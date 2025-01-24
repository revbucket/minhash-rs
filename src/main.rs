
use std::collections::HashSet;
use std::io::Write;
use std::fs::OpenOptions;
use std::os::unix::fs::OpenOptionsExt;
use glob::glob;
use std::cmp::max;
use std::sync::Mutex;
use serde::{Serialize, Deserialize};
use std::collections::{VecDeque, HashMap};
use std::hash::{Hash, Hasher, DefaultHasher};
use anyhow::{Result, Error, anyhow};
use std::path::{PathBuf};
use std::io::{BufRead};
use rand::prelude::*;
use tiktoken_rs::{p50k_base, CoreBPE};
use serde_json::{Value, json};
use regex::Regex;
use ndarray::{Array1};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sha2::{Sha256, Digest};
use dashmap::{DashMap, DashSet};
use rayon::prelude::*;
use clap::{Parser, Subcommand};
use crate::io::{expand_dirs, read_pathbuf_to_mem, write_mem_to_pathbuf, get_output_filename, has_json_extension};
use crate::storage::{IntValueEnum, SignatureWriter, GenWriter, MinHashConfig, FileMap, to_byte_size, compute_sig_size};
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Instant;
use std::sync::atomic::{AtomicUsize, Ordering};
use bincode;

use crate::uf_rush2::UFRush;
use std::panic::catch_unwind;
use std::fs::create_dir_all;
use std::option::Option;

pub mod s3;
pub mod io;
pub mod storage;
pub mod uf_rush2;


const BIG_PRIME: u64 = 18446744073709551557;

const MERSENNE_PRIME: u64 = (1 << 61) - 1;
const MAX_HASH: u64 = (1 << 32) - 1;
const CC_CHUNK_SIZE: usize = usize::MAX; //100 * 1024 * 1024; // 100 MB chunk size


/*
New plan:
It helps to consider minHash in phases:
Phase 1: Compute (band_id, signature, doc_id) for every document
Phase 2: For each band, group along signatures and take all but the first doc_id 
         in each group to add to a 'lines to kill' set (file_id -> {lines...})
Phase 3: Merge all the 'lines to kill' above 
Phase 4: Delete lines from documents and put to outputs

Where the slowest step (by far!) is phase 1. This also requires a bunch of RAM

It helps to think of phase one as a matrix where rows
                
                        Files
             ----------------------------
            |                            |
            |                            |
      Bands |                            |
            |                            |
            |                            |
            ------------------------------
Where the entries are signatures.

Plan:
Step 1:  Build and save PathLookup object 
Step 2:  Break matrix into submatrices groups of rows/cols, handle each submatrix separately.
         For a submatrix (group of files, group of band_seeds):
         - Loop over files (in parallel)
         - For each file:
             + For each document in file:
                 * Compute signature for all bands in this submatrix
                 * Write to a file ON DISK with file structure like:
                     band_id/                        # which band id/band seed this is (which row)
                         sigchunk_0000/              # chunk signatures based on range (maybe first byte?)
                             filechunk_0000.bin      # which group of files this is    (which cols)
                 where each line gets a bytestring of (signature,file_id,line_num)
         - And then either:
             + upload all these files
             + proceed through all submatrices

Step 3:  Merge all ^ files computed above together into a to-kill-list
         - Maintain a global to-kill-list structure
         - For each band_id/sigchunk:
             - Group the lines together in lists (dashmap<signature, Vec<doc_id>>)
             - Add all but the first element of each group into global structure 

Step 4: Use global to-kill-list to clean dataset of duplicates 


NOTE:
    general pattern for async-y things is that we will init a runtime 
    to interact with s3 (read, write ,list files) and block until that's done. 
*/



/*=================================================================
=                                  ARGS                           =
=================================================================*/

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct ArgParser {
    #[clap(subcommand)]
    command: Commands,

    #[arg(long, default_value_t=0)]
    threads: usize,
}

#[derive(Subcommand, Debug)]
enum Commands {
    #[clap(arg_required_else_help = true)]

    MinHash { 
        /// (List of) directories/files (on s3 or local) that are jsonl.gz or jsonl.zstd files
        #[arg(required=true, long, num_args=1..)]
        input: Vec<PathBuf>,

        /// Output location (may be an s3 uri)
        #[arg(required=true, long)]
        output: PathBuf,

        #[arg(long, default_value_t=10_000_000_000)] // 10B docs by default?
        num_docs: usize,

        #[arg(long, default_value_t=16_000_000)] // 3 bytes by default
        max_lines_per_path: usize,

        #[arg(required=true, long)]
        sig_storage: PathBuf,


        #[arg(long, default_value_t=14)]
        num_bands: u32,

        #[arg(long, default_value_t=9)]
        band_size: usize,    

        #[arg(long, default_value_t=5)]
        ngram_size: usize,  
     }, 


    BuildFileMap {
        #[arg(required=true, long)]
        config: PathBuf   
    },

    BuildConfig {
        // Just makes and saves the path lookup object 

        /// Input locations for paths to hash
        #[arg(required=true, long, num_args=1..)]
        input: Vec<PathBuf>,        

        /// Output location (may be an s3 uri)
        #[arg(required=true, long)]
        output: PathBuf,

        /// How many documents we have 
        /// (used to infer how many bytes we need to store)
        #[arg(long, default_value_t=10_000_000_000)] // 10B docs by default?
        num_docs: usize,

        /// Max # of lines per document 
        /// (used to infer how many bytes we need to store)
        #[arg(long, default_value_t=16_000_000)] // 3 bytes by default
        max_lines_per_path: usize,
    },

    HashOnly {
        // Just runs and saves the hashes
        #[arg(required=true, long)]
        config: PathBuf,

        #[arg(long, default_value_t=0)]
        path_chunk: usize,

        #[arg(long, default_value_t=1)]
        num_path_chunks: usize
    },


    GatherEdges {
        // Saves the "edge" objects
        #[arg(required=true, long)]
        config: PathBuf
    },

    Finish {
        /// Path name of the config file
        #[arg(required=true, long)]
        config: PathBuf,

        /// where the stored hashes live
        #[arg(required=true, long)]
        sig_storage: PathBuf,

        /// Where the output files go 
        #[arg(required=true, long)]
        output: PathBuf,        

    },

    BuildEdges {
        #[arg(required=true, long)]
        config: PathBuf,

        #[arg(required=true, long)]
        sig_storage: PathBuf,

        #[arg(required=true, long)]
        group_storage: PathBuf,
    },


    BuildUf {
        // Gets a mapping from (path_id, line_num) -> cc_hash
        // Where the hash identifies the connected component
        #[arg(required=true, long)]
        config: PathBuf,

        #[arg(long, default_value_t=1)]
        num_path_chunks: usize,
    },


    UfSizePrune {
        // Prunes a dataset based on the connected component sizes:
        #[arg(required=true, long)]
        config: PathBuf, 

        #[arg(required=true, long)]
        ccs: PathBuf,

        #[arg(required=true, long)]
        output: PathBuf,

        /// Removes all documents that belong to a cc that has size <= than this
        #[arg(required=true, long)]
        floor: usize,

        /// Of the ccs with size > floor, only keeps this many of the ccs
        #[arg(required=true, long)]
        ceil:usize
    },

    AnnotateCc {
        // Annotates each document with their CC size 

        #[arg(required=true, long)]
        config: PathBuf,    
        #[arg(required=true, long)]
        ccs: PathBuf, 
        #[arg(required=true, long)]
        output: PathBuf,      
    },             

    CountDocsFromSigs {
        #[arg(required=true, long)]
        config: PathBuf,

        #[arg(required=true, long)]
        sig_storage: PathBuf,
    }

}

/*=================================================================
=                             UTILITIES                           =
=================================================================*/


fn build_pbar(num_items: usize, units: &str) -> ProgressBar {
    let mut template = String::from(units);
    template.push_str(" {human_pos}/{human_len} [{elapsed_precise}/{duration_precise}] [{wide_bar:.cyan/blue}]");
    let pbar = ProgressBar::new(num_items as u64)
        .with_style(
            ProgressStyle::with_template(&template).unwrap()
        );
    pbar.inc(0);
    pbar
}



/*=================================================================
=                          PROCESS SINGLE FILE                    =
=================================================================*/
/* 
Input in this section is a single file (pathbuf), all hyperparams,
and the shared data structure keeping track mapping:
    {(band_seed) -> {signature -> [(path_id, line_id), ...]}}
Preprocessing flow is to use the slimpajama flow, and then tokenize with tiktoken
*/ 


fn process_path(path: &PathBuf, band_seeds: &Vec<u32>, path_id: usize, band_size: usize, ngram_size: usize,
                 tokenizer_str: &str, signature_writer: &SignatureWriter, num_sig_chunks: usize,
                 path_size: usize, line_size: usize, sig_size: usize) -> Result<usize, Error> {
    // Setup things: load data, build tokenizer, etc
    let data = read_pathbuf_to_mem(path).unwrap();
    // let mut buffer = Vec::new();
    // data.read_to_end(&mut buffer).unwrap();
    // println!("READ DATA {:?}", buffer);
    let tokenizer = load_tokenizer(tokenizer_str).unwrap();
    let num_bands = band_seeds.len();
    let perm_seeds = _expand_band_seeds(&band_seeds, band_size);
    let path_id = IntValueEnum::new(path_id, path_size);
    let mut docs_hashed = 0;
    for (line_num, line) in data.lines().enumerate() {

        let line_num = IntValueEnum::new(line_num, line_size);
        let line = line.unwrap();
        docs_hashed += 1;
        let json: Value = serde_json::from_str(&line).expect(&format!("Failed to parse {:?} {:?}", path.clone(), line_num.as_usize()));
        let text = json["text"].as_str().unwrap();

        let Ok(tokens) = catch_unwind(|| preprocess_text(text, &tokenizer)) else {
            println!("Tokenization failed on {:?} | {:?} | {:?}", path.clone(), path_id, line_num);
            continue;
        };
        let hash_vals = get_hash_vals_from_tokens(tokens, &perm_seeds, ngram_size);
        let bands = hash_vals.into_shape((num_bands, band_size)).unwrap();
        for (row, band_seed) in bands.rows().into_iter().zip(band_seeds.iter()) {
            let mut hasher = Sha256::new(); 
            hasher.update(bytemuck::cast_slice(row.as_slice().unwrap()));
            let hash = hasher.finalize();
            let band_signature = IntValueEnum::from_bytes(hash[..sig_size].to_vec(), sig_size);   
            _save_band_signature_to_disk(&signature_writer, *band_seed, band_signature, path_id.clone(), line_num.clone(), num_sig_chunks).unwrap();
        }
    }
    Ok(docs_hashed)
}


fn load_tokenizer(tokenizer_name: &str) -> Result<CoreBPE, Error> {
    match tokenizer_name {
        "p50k" => Ok(p50k_base().unwrap()),
        _ => panic!("Unknown tokenizer {:?}", tokenizer_name)
    }
}

fn preprocess_text(text: &str, tokenizer: &CoreBPE) -> Vec<usize> {
    // Clean text and then tokenize
    let text = clean_text(text);
    tokenizer.encode_with_special_tokens(&text)
}


fn clean_text(text: &str) -> String {
    // SlimPajama text cleaning process

    // Convert the document to lowercase
    let mut text = text.to_lowercase();

    // Remove punctuation
    let punctuation: &[_] = &['!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~'];
    text.retain(|c| !punctuation.contains(&c));

    // Replace multiple whitespace characters with a single space
    let re = Regex::new(r"\s+").unwrap();
    text = re.replace_all(&text, " ").to_string();

    // Trim leading and trailing whitespace
    text.trim().to_string()
}

fn get_hash_vals_from_tokens(tokens: Vec<usize>, perm_seeds: &Vec<u64>, ngram_size: usize) -> Array1<u64> {
    let (a,b) = _init_permutations(perm_seeds);
    let n = perm_seeds.len();
    let mut hash_vals = Array1::ones(n) * MAX_HASH;
    let mut ngram: VecDeque<usize> = VecDeque::with_capacity(ngram_size);
    let mut ngram_count = 0; 
    for token in tokens {
        ngram.push_back(token);
        if ngram.len() >= ngram_size {
            ngram_count += 1;
            hash_vals = _update_hash_vals(hash_vals, &a, &b, &ngram);
            ngram.pop_front();
        }
    }
    hash_vals = if ngram_count == 0 {
        _update_hash_vals(hash_vals, &a, &b, &ngram) // short document, still wanna hash it
    } else {
        hash_vals
    };

    hash_vals
}
    


fn _init_permutations(seeds: &Vec<u64>) -> (Array1<u64>, Array1<u64>) {
    // Initialize the permutations needed for each minhash
    let n = seeds.len();
    let mut a = Array1::zeros(n);
    let mut b = Array1::zeros(n);    
    for (i, &seed) in seeds.iter().enumerate() {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        a[i] = rng.gen();
        b[i] = rng.gen();
    }
    (a,b)    
}


fn _update_hash_vals(mut hash_vals: Array1<u64>, a: &Array1<u64>, b: &Array1<u64>, ngram: &VecDeque<usize>) -> Array1<u64> {
    // hash ngram and do the minhash update
    let mut hasher = DefaultHasher::new();
    ngram.hash(&mut hasher);
    let cur_hash = hasher.finish();
    let mut phv = a.clone();
    // next line is: (a * cur_hash + b) % P [wrap on overflow]
    phv.zip_mut_with(&b, |x, y| *x = mod_add(mod_mul(*x, cur_hash, BIG_PRIME), *y, BIG_PRIME));
    hash_vals.zip_mut_with(&phv, |x, y| *x = std::cmp::min(*x, *y));
    hash_vals

}

fn _expand_band_seeds(band_seeds: &Vec<u32>, band_size: usize) -> Vec<u64> {
    // Each "band seed" is expanded here to band_size random u64s, and flattened. (used to seed permutations)
    // Probably like no collisions here, so let's just not worry about that ;) 

    let mut perm_seeds: Vec<u64> = Vec::new();
    for band_seed in band_seeds.iter() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(*band_seed as u64);
        for _i in 0..band_size {
            perm_seeds.push(rng.next_u64());
        }
    }
    perm_seeds
}

fn _save_band_signature_to_disk(signature_writer: &SignatureWriter, band_seed: u32, band_signature: IntValueEnum, 
                                path_id: IntValueEnum, line_num: IntValueEnum, num_sig_chunks: usize) -> Result<(), Error> {

    let sig_chunk = band_signature.as_usize() % num_sig_chunks;
    let contents = [band_signature.as_bytes(), path_id.as_bytes(), line_num.as_bytes()].concat();
    signature_writer.write_line(band_seed, sig_chunk, contents).unwrap();
    Ok(())
}

fn mod_mul(a: u64, b: u64, modulus: u64) -> u64 {
    let mut result = 0;
    let mut a = a % modulus;
    let mut b = b % modulus;
    
    while b > 0 {
        if b & 1 != 0 {
            result = (result + a) % modulus;
        }
        a = (a << 1) % modulus;
        b >>= 1;
    }
    
    result
}

fn mod_add(a: u64, b: u64, modulus: u64) -> u64 {
    let a = a % modulus;
    let b = b % modulus;
    
    // Add and take modulus
    // If a + b would overflow, we can do it this way:
    if modulus - a <= b {
        // If we get here, then a + b would wrap around the modulus
        // So we calculate: (a + b) mod m = (a - (m - b)) mod m
        let difference = b - (modulus - a);
        difference % modulus
    } else {
        // Normal case where a + b doesn't wrap
        (a + b) % modulus
    }
}


/*=================================================================
=                            EDGE GATHERING HELPERS               =
=================================================================*/

fn gather_groups(sig_storage: PathBuf) -> Result<DashMap<(usize, usize), Vec<PathBuf>>, Error> {
    let binding = sig_storage.clone().join("**").join("*.sig.bin");
    let sig_storage_str = binding.to_str().unwrap();
    let map: DashMap<(usize, usize), Vec<PathBuf>> = DashMap::new();
    for entry in glob(&sig_storage_str).unwrap() {
        let entry = entry.unwrap();
        let sigchunk_dir = entry
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|name| name.to_str())
            .unwrap();
        let sigchunk = sigchunk_dir.split('_').last().unwrap().parse::<usize>().unwrap();
    
        let band_id_dir = entry
            .parent().unwrap().parent()
            .and_then(|p| p.file_name())
            .and_then(|name| name.to_str())
            .unwrap();
        let band_id = band_id_dir.split('_').last().unwrap().parse::<usize>().unwrap();

        map.entry((sigchunk, band_id)).or_default().push(entry);
    }    

    Ok(map)
}






/*=================================================================
=                      COLLECT DOCS TO DELETE                     =
=================================================================*/

fn aggregate_lines_to_kill(config: &MinHashConfig, sig_storage: &PathBuf) -> 
    Result<DashMap<usize, DashSet<usize>>, Error> {
    println!("Aggregating hashes into lines to kill...");
    let start_main = Instant::now();

    // Gather all files in storage and group by (band, sigchunk)
    let band_sigs = _collect_band_sigs(sig_storage).unwrap();

    let lines_to_kill : DashMap<usize, DashSet<usize>> = DashMap::new();
    let lines_to_kill_pbar = build_pbar(band_sigs.len(), "File groups");
    band_sigs.par_iter()
        .for_each(|v| {
            _augment_lines_to_kill(&lines_to_kill, &v, config.sig_size, config.path_size, config.line_size).unwrap();
            lines_to_kill_pbar.inc(1);
        });

    println!("-------------------------");
    println!("Aggregated all lines to kill in {:?}", start_main.elapsed().as_secs());
    Ok(lines_to_kill)
}


fn _collect_band_sigs(sig_storage: &PathBuf) -> Result<DashMap<(u32, usize), Vec<PathBuf>>, Error> {
    let band_sigs : DashMap<(u32, usize), Vec<PathBuf>> = DashMap::new();
    let all_files = expand_dirs(vec![sig_storage.clone()], Some(vec![".sig.bin"].as_slice())).unwrap();
    all_files.par_iter()
        .for_each(|p| {
            let key = _extract_bandid_sigchunk(p).unwrap();
            band_sigs.entry(key).or_default().push(p.clone());
        });
    Ok(band_sigs)
}
    
fn _extract_bandid_sigchunk(path: &PathBuf) -> Result<(u32, usize), Error> {
    let re = Regex::new(r"band_(\d+)/sigchunk(\d+)").unwrap();
    if let Some(path_str) = path.to_str() {
        if let Some(captures) = re.captures(path_str) {
            let band_number = captures[1].parse::<u32>().unwrap();
            let sigchunk_number = captures[2].parse::<usize>().unwrap();
            return Ok((band_number, sigchunk_number));
        }
    }
    Err(anyhow!("Failed to extract band_id/sig_chunk!"))
}

fn _augment_lines_to_kill(lines_to_kill: &DashMap<usize, DashSet<usize>>, paths: &Vec<PathBuf>, 
                          sig_size: usize, path_size: usize, line_size: usize) -> Result<(), Error> {
    let aug_start = Instant::now();
    // Load all data and create mapping of {sig -> [(doc_id, line_num),...]}
    let entry_size = sig_size + path_size + line_size;
    let cur_set : DashSet<IntValueEnum> = DashSet::new();
    paths.iter().for_each(|path| {
        let contents = read_pathbuf_to_mem(path).unwrap().into_inner().into_inner();
        contents.par_chunks(entry_size).for_each(|entry| {
            let sig = IntValueEnum::from_bytes(entry[..sig_size].to_vec(), sig_size);
            let path_id = IntValueEnum::from_bytes(entry[sig_size..sig_size+path_size].to_vec(), path_size);
            let line_id = IntValueEnum::from_bytes(entry[sig_size+path_size..].to_vec(), line_size);
            let newly_inserted = cur_set.insert(sig);
            if !newly_inserted {
                lines_to_kill.entry(path_id.as_usize()).or_default().insert(line_id.as_usize());                
            }
        });
    });
    //println!("(Aug) Loaded path data into groups in {:?} secs", aug_start.elapsed().as_secs());
    return Ok(());
}

/*=================================================================
=                      UNION FIND HELPERS                         =
=================================================================*/

#[derive(Serialize, Deserialize)]
struct BandGroup(Vec<Vec<(usize, usize)>>);


fn build_band_group(band_sigs: &Vec<PathBuf>, sig_size: usize, path_size: usize, line_size: usize, 
                    singleton_map_opt: &Option<DashMap<usize, usize>>) -> 
    Result<Vec<Vec<(usize, usize)>>, Error> {
    // For a group of files that contain signatures within the same band (and a sig chunk)
    // Collects a list of (path_id: usize, line_id: usize) for each clique
    // (reading each file is done in parallel, so nothing upstream should be par_iter'ed)

    let entry_size = sig_size + path_size + line_size;

    // build map from signature -> [(path_id, line_num), ...]
    // to collect docs that have the same signature within this band
    let group_map : DashMap<IntValueEnum, Vec<(usize, usize)>> = DashMap::new();


    band_sigs.iter().for_each(|path| {
        let contents = read_pathbuf_to_mem(path).unwrap().into_inner().into_inner();
        contents.chunks(entry_size).for_each(|entry| {
            let sig = IntValueEnum::from_bytes(entry[..sig_size].to_vec(), sig_size);
            let path_id = IntValueEnum::from_bytes(entry[sig_size..sig_size+path_size].to_vec(), path_size).as_usize();
            let line_id = IntValueEnum::from_bytes(entry[sig_size+path_size..].to_vec(), line_size).as_usize();
            group_map.entry(sig).or_default().push((path_id, line_id));

            if let Some(singleton_map) = singleton_map_opt {
                singleton_map.entry(path_id).and_modify(|cur_max| {
                    if line_id > *cur_max {
                        *cur_max = line_id;
                    }
                }).or_insert(line_id);
            }
        });
    });
    
    // Select only the groups that have size > 1
    let band_group: Vec<Vec<(usize, usize)>> = group_map
        .into_iter()
        .map(|(_, group)| group)
        .filter(|value| value.len() >1)
        .collect();

    Ok(band_group)
}

fn save_band_group2(band_group: Vec<Vec<(usize, usize)>>, output_file: PathBuf, path_size: usize, line_size: usize) -> Result<(), Error> {
    let max_path = IntValueEnum::new(1 << path_size - 1, path_size);
    let max_line = IntValueEnum::new(1 << line_size - 1, line_size);
    let group_end = [max_path.as_bytes(), max_line.as_bytes()].concat();


    if let Some(parent_dir) = output_file.parent() {
        if !parent_dir.exists() {
            create_dir_all(parent_dir).unwrap();
        }
    }
    let mut writer = OpenOptions::new()
                 .append(true)
                 .create(true)
                 .mode(0o644)
                 .open(output_file)
                 .unwrap();    

    band_group.into_iter()
        .for_each(|edgelist| {
            edgelist.into_iter()
                .for_each(|(path_id, line_num)| {
                    let contents = [IntValueEnum::new(path_id, path_size).as_bytes(),
                                    IntValueEnum::new(line_num, line_size).as_bytes()]
                                    .concat();
                    writer.write_all(&contents).unwrap();
        });
        writer.write(&group_end.clone()).unwrap();
    });
    writer.flush().unwrap();
    Ok(())
}
    


fn save_band_group(band_group: Vec<Vec<(usize, usize)>>, output: &PathBuf, band_id: u32, sig_chunk: usize) 
    -> Result<(), Error> {
    let band_group_name = _get_band_group_name(output, band_id, sig_chunk);
    let serialized: Vec<u8> = serde_json::to_vec(&band_group).unwrap();//::serialize(&BandGroup(band_group)).unwrap();

    write_mem_to_pathbuf(&serialized, &band_group_name)
}


fn _get_band_group_name(band_group_storage: &PathBuf, band_id: u32, sig_chunk: usize) -> PathBuf {
    band_group_storage.clone()
        .join(format!("band_{:016}", band_id))
        .join(format!("bandgroup{:08}.group.bin.gz", sig_chunk))
}

fn add_edge_file_to_uf(edge_file: &PathBuf, uf: &UFRush, path_size: usize, line_size: usize) -> Result<(), Error> {
    let edge_data = read_pathbuf_to_mem(edge_file).unwrap().into_inner().into_inner();
    let max_path = IntValueEnum::new(1 << path_size - 1, path_size).as_usize();
    let max_line = IntValueEnum::new(1 << line_size - 1, line_size).as_usize();
    let group_end_id = pair2docid((max_path, max_line), line_size);
    let mut last_id = group_end_id;

    edge_data.chunks_exact(path_size + line_size).for_each(|c| {
        let path_id = IntValueEnum::from_bytes(c[..path_size].to_vec(), path_size).as_usize();
        let line_num = IntValueEnum::from_bytes(c[path_size..].to_vec(), line_size).as_usize();
        let cur_id = pair2docid((path_id, line_num), line_size);
        if cur_id != group_end_id && last_id != group_end_id {
            uf.unite(last_id, cur_id);
        }
        last_id = cur_id;
    });

    Ok(())
}


fn add_band_group_to_uf(band_group: &PathBuf, uf: &UFRush, config: &MinHashConfig) -> Result<(), Error> {
    // Adds all groups in the band group to the unionfind
    let line_size = config.line_size;
    let serialized = read_pathbuf_to_mem(band_group).unwrap().into_inner().into_inner();

    let band_group: Vec<Vec<(usize, usize)>> = serde_json::from_slice(&serialized).unwrap();
    
    band_group
        .into_iter()
        .for_each(|mut group| {
            let last = group.pop().unwrap();
            let last_id = pair2docid(last, line_size);            
            while group.len() > 0 {
                let cur = group.pop().unwrap();
                let cur_id = pair2docid(cur, line_size);
                uf.unite(last_id, cur_id);
            }
    });
    Ok(())
}

fn build_ccs(uf: UFRush, line_size: usize) -> Vec<((usize,usize), usize)> {
    let size = uf.nodes.len();

    let keys: Vec<usize> = uf.nodes.par_iter().map(|entry| *entry.key()).collect();
    println!("LEN KEYS IS {:?}", keys.len());
    println!("LINE SIZE IS {line_size}");
    let pbar = build_pbar(size, "Docs");
    keys.into_par_iter()
    .map(|key| {
        pbar.inc(1);
        (docid2pair(key, line_size), uf.find(key))
    })
    .collect()
}


fn pair2docid(pair: (usize, usize), line_size: usize) -> usize {
    // Given a (path_id, line_id) pair, converts it into a single usize 
    // (which is needed for UF rush)
    let (path_id, line_id) = pair;
    (path_id << (line_size * 8) ) + line_id
}

fn docid2pair(docid: usize, line_size: usize) -> (usize, usize) {
    // Inverse function of the pair2docid
    let mask = (1 << (line_size * 8)) - 1;
    (docid >> (line_size * 8), docid & mask)

}

fn count_cc_sizes(ccs: Vec<((usize, usize), usize)>) -> HashMap<usize, usize> {
    // First map only the cc hashes into a dashmap counting them 
    let ccid2count : DashMap<usize, usize> = DashMap::new();
    let cc_counter_pbar = build_pbar(ccs.len(), "Docs");

    ccs.par_iter()
        .for_each(|(_, cc_id)| {
            ccid2count.entry(*cc_id).and_modify(|e| *e += 1).or_insert(1);
            cc_counter_pbar.inc(1);
        });

    // Then map values into a second dashmap counting sizes 
    let cc_sizes_dashmap: DashMap<usize, usize> = DashMap::new();
    let cc_sizes_pbar = build_pbar(cc_sizes_dashmap.len(), "CCS");
    ccid2count.par_iter().for_each(|entry| {
        let cc_size = *entry.value();
        cc_sizes_dashmap.entry(cc_size).and_modify(|e| *e +=1).or_insert(1);
        cc_sizes_pbar.inc(1);
    });

    // Then return a hashmap of these things:
    let cc_sizes : HashMap<usize, usize> = cc_sizes_dashmap
        .into_iter()
        .map(|(k, v)| (k,v)).collect();
    cc_sizes
}

fn collect_singletons(band_sigs: &Vec<PathBuf>, max_lines_per_doc: &DashMap<usize, usize>, config: &MinHashConfig) -> Result<(), Error> {
    // band_sigs is files with signature entries
    // max_lines_per_doc maps {doc_id: max(line_nums)} [mutated but not returned]
    let pbar = build_pbar(band_sigs.len(), "Band Sig Files");
    let entry_size = config.sig_size + config.path_size + config.line_size;
    let sig_size = config.sig_size;
    let line_size = config.line_size;
    let path_size = config.path_size;

    band_sigs.par_iter().for_each(|path| {
        let contents = read_pathbuf_to_mem(path).unwrap().into_inner().into_inner();
        contents.chunks(entry_size).for_each(|entry| {
            let path_id = IntValueEnum::from_bytes(entry[sig_size..sig_size+path_size].to_vec(), path_size).as_usize();
            let line_id = IntValueEnum::from_bytes(entry[sig_size+path_size..].to_vec(), line_size).as_usize();
            max_lines_per_doc.entry(path_id).and_modify(|e| *e = max(*e, line_id)).or_insert(line_id);
        });
        pbar.inc(1)
    });    


    //println!("(Aug) Loaded path data into groups in {:?} secs", aug_start.elapsed().as_secs());
    Ok(())
}




fn save_singletons(max_lines_per_doc: DashMap<usize, usize>, singleton_path: &PathBuf) 
    -> Result<(), Error> {
    let data: Vec<u8> = max_lines_per_doc
        .par_iter()
        .flat_map(|r| {
            let key_bytes: u64 = (*r.key()) as u64;
            let val_bytes: u64 = (*r.value()) as u64;
            vec![key_bytes.to_le_bytes(), val_bytes.to_le_bytes()]
        })
        .flat_map(|a| a)
        .collect();        

    write_mem_to_pathbuf(&data.as_slice(), singleton_path);
    Ok(())    
}

fn load_singletons(singleton_path: &PathBuf) -> Result<Vec<(usize, usize)>, Error> {
    let data = read_pathbuf_to_mem(singleton_path).unwrap().into_inner().into_inner();

    let singletons: Vec<(usize, usize)> = data.chunks_exact(16).map(|c| {
        let path_id = u64::from_le_bytes(c[..8].try_into().unwrap()) as usize;
        let line_num = u64::from_le_bytes(c[8..].try_into().unwrap()) as usize;
        (path_id, line_num)
    }).collect();

    Ok(singletons)
    
}


fn add_singletons_to_uf(singletons: Vec<(usize, usize)>, uf: &UFRush, line_size: usize) -> Result<(), Error> {
    let pbar = build_pbar(singletons.len(), "Singleton paths");
    singletons.par_iter().for_each(|(path_id, max_line)| {
        for line_num in 0..(max_line+1) {
            let cur_id = pair2docid((*path_id, line_num), line_size);
            uf.find(cur_id);
        }
        pbar.inc(1);
    });
    Ok(())
}


fn save_all_ccs(cc_map: &DashMap<usize,Vec<(usize, usize)>>, cc_dir: &PathBuf, num_chunks: usize) -> Result<(), Error> {
    
    let writer = GenWriter::new(cc_dir, num_chunks, "cc");    
    let pbar = build_pbar(cc_map.len(), "Saving CCs");
    let max_u64_bytes = u64::MAX.to_le_bytes();

    let terminus: Vec<u8> = vec![max_u64_bytes, max_u64_bytes].into_iter().flat_map(|a| a).collect();
    cc_map.par_iter().for_each(|entry| {
        // Create contents before locking 
        let key = entry.key();
        let value = entry.value();
        if value.len() > 1 {
            let mut val_bytes: Vec<u8>= value.iter().flat_map(|r| {
                vec![(r.0 as u64).to_le_bytes(), 
                     (r.1 as u64).to_le_bytes()]
            }).flat_map(|a| a)
            .collect();
            val_bytes.extend(terminus.clone());
            writer.write_line(*key, val_bytes, 0).unwrap();
        }
        pbar.inc(1);
    });

    writer.finish().unwrap();
    Ok(())
}


fn collect_kill_list(cc_map: DashMap<usize, Vec<(usize, usize)>>) -> DashMap<usize, Vec<usize>> {
    // Creates map of path_id -> [line_num,...] for lines to remove from doc 
    let pbar = build_pbar(cc_map.len(), "Organizing kill ccs");
    let kill_list: DashMap<usize, Vec<usize>>  = DashMap::new();
    cc_map.par_iter().for_each(|entry| {
        let value: &Vec<(usize, usize)> = entry.value();
        if value.len() > 1 {
            for i in 0..value.len() - 1 {
                let (path_id, line_num) = value[i];
                kill_list.entry(path_id).or_default().push(line_num);
            }
        }
        pbar.inc(1);
    }); 

    kill_list
}


fn save_kill_list(kill_list: DashMap<usize, Vec<usize>>, kill_dir: &PathBuf, file_map: &FileMap, num_path_chunks: usize) -> Result<(), Error> {

    // Map path id to chunk id
    let path_id_2_chunk_id : DashMap<usize, usize> = DashMap::new();
    for chunk_id in 0..num_path_chunks {
        let path_chunk = file_map.get_path_chunk(chunk_id, num_path_chunks);
        path_chunk.par_iter().for_each(|entry| {
            let path_id = entry.1;
            path_id_2_chunk_id.insert(path_id, chunk_id);
        });        
    }

    // Then loop through kill list and write as we go:

    let writer = GenWriter::new(kill_dir, num_path_chunks, "kill");
    let pbar = build_pbar(kill_list.len(), "Writing kill list");
    let terminus = u64::MAX.to_le_bytes();
    kill_list.par_iter().for_each(|entry| {
        let path_id = entry.key();
        let chunk_id = path_id_2_chunk_id.get(path_id).unwrap();
        let line_nums: &Vec<usize> = entry.value();

        let mut contents : Vec<u8>  = Vec::new();
        contents.extend(path_id.to_le_bytes());
        let mut lines_concat: Vec<u8> = line_nums.into_iter().map(|i| i.to_le_bytes()).flat_map(|a| a).collect();
        lines_concat.extend(terminus.clone());
        contents.extend(lines_concat);
        writer.write_line(0, contents, *chunk_id).unwrap();
        pbar.inc(1);
    });
    writer.finish().unwrap();
    Ok(())
}


fn parse_kill_file(kill_file: &PathBuf) -> Result<DashMap<usize, Vec<usize>>, Error> {
    let kill_list: DashMap<usize, Vec<usize>> = DashMap::new();

    let contents = read_pathbuf_to_mem(kill_file).unwrap().into_inner().into_inner();
    let mut path_id: usize = 0;
    let mut restart = true;

    let terminus = u64::MAX.to_le_bytes();

    for chunk in contents.chunks_exact(8) {
        if chunk == terminus {
            restart = true;
            continue;
        }
        else if restart == true {
            path_id = u64::from_le_bytes(chunk.try_into().unwrap()) as usize;
            restart = false;
        } else {
            let line_num: usize = u64::from_le_bytes(chunk.try_into().unwrap()) as usize;
            kill_list.entry(path_id).or_default().push(line_num);
        }
    }

    Ok(kill_list)
}


/*=================================================================
=                         UF SIZE PRUNE HELPERS                   =
=================================================================*/

fn load_cc_list(cc_dir: &PathBuf) -> Result<Vec<((usize, usize), usize)>, Error> {
    // Takes the CC directory and gathers all the files, loads them and then groups objects
    // according to which CC they belong to 
    let cc_ext = ".cc.bin";
    let mut cc_paths = expand_dirs(vec![cc_dir.clone()], Some(&[&cc_ext])).unwrap();
    cc_paths.sort();
    
    // Read the chunks into memory 
    let data_read_pbar = build_pbar(cc_paths.len(), "CC Chunks (loading)");
    let cc_chunks: Vec<Vec<u8>> = cc_paths.par_iter().map(|p| {
        let chunk_contents = read_pathbuf_to_mem(&p).unwrap().into_inner().into_inner();
        data_read_pbar.inc(1);
        chunk_contents
    }).collect();

    let mut cc_bytes = Vec::new();
    let cat_pbar = build_pbar(cc_chunks.len(), "CC Chunks (concat2)");
    for chunk in cc_chunks {
        cc_bytes.extend(chunk);
        cat_pbar.inc(1);
    }

    let ccs = bincode::deserialize(&cc_bytes).unwrap();
    Ok(ccs)
}  


fn group_docs_by_cc(cc: &Vec<((usize, usize), usize)>) -> DashMap<usize, Vec<(usize, usize)>> {
    // Maps [((doc_id, line_num), cc_id)] -> {cc_id -> [(doc_id, line_num), ...]}
    let cc_group: DashMap<usize, Vec<(usize, usize)>> = DashMap::new();

    let cc_pbar = build_pbar(cc.len(), "Docs");
    cc.into_par_iter()
        .for_each(|((doc_id, line_num), cc_id)| {
            cc_group.entry(*cc_id).or_default().push((*doc_id, *line_num));
            cc_pbar.inc(1);
        }
    );
    cc_group
}

fn gather_lines_to_live(cc_groups: DashMap<usize, Vec<(usize, usize)>>, floor: usize, ceil: usize) -> DashMap<usize, DashSet<usize>> {
    // Gathers a collection of (doc_id, line_num) that SURVIVE given the floor and ceil
    let survivor_pbar = build_pbar(cc_groups.len(), "CCs");
    let survivors : Vec<(usize, usize)> = cc_groups
        .into_par_iter()
        .flat_map(|(_, val)| {
            let survivors: Vec<(usize, usize)> = if val.len() <= floor {
                // If cc is <= floor, keep nothing
                Vec::new()
            } else if val.len() <= ceil {
                // If cc is > floor but <= ceil, keep everything
                val
            } else {
                // otherwise randomly select ceil to keep
                let mut rng = thread_rng();
                val.choose_multiple(&mut rng, ceil).cloned().collect()
            };
            survivor_pbar.inc(1);
            survivors
        }).collect();

    // And then group by doc id
    let grouped_survivors : DashMap<usize, DashSet<usize>> = DashMap::new();
    let group_pbar = build_pbar(survivors.len(), "Survivors");
    survivors.into_par_iter()
        .for_each(|(doc_id, line_num)| {
            grouped_survivors.entry(doc_id).or_default().insert(line_num);
            group_pbar.inc(1);
        });
    grouped_survivors
}

fn gather_lines_to_kill_ccs(cc_groups: DashMap<usize, Vec<(usize, usize)>>, floor:usize, ceil:usize) -> 
    DashMap<usize, DashSet<usize>> 
{
    // Gathers a collection of (doc_id, line_num) that should get REMOVED, given the ceil
    assert!(floor == 0);
    let ltk_pbar = build_pbar(cc_groups.len(), "CCs");
    let grouped_ltk : DashMap<usize, DashSet<usize>> = DashMap::new();
    cc_groups
        .into_par_iter()
        .for_each(|(_, v)| {
            let ltk: Vec<(usize, usize)> = if v.len() < ceil {
                v
            } else {
                let mut rng = thread_rng();
                v.choose_multiple(&mut rng, v.len() - ceil).cloned().collect::<Vec<(usize, usize)>>()
            };
            for (doc_id, line_num) in ltk {
                grouped_ltk.entry(doc_id).or_default().insert(line_num);
            }
            ltk_pbar.inc(1);
        }
    );
    grouped_ltk
}

/*=================================================================
=                         ANNOTATE HELPERS                        =
=================================================================*/

fn build_annotation_lookup(cc_groups: DashMap<usize, Vec<(usize, usize)>>) ->  DashMap<usize, DashMap<usize, usize>> {
    // Maps {cc_id -> [(doc_id, line_num),...]} to {doc_id -> {line_id: cc_size}}
    let annotations : DashMap<usize, DashMap<usize, usize>> = DashMap::new();
    let pbar = build_pbar(cc_groups.len(), "CCs");

    cc_groups.into_par_iter()
        .for_each(|(cc_id, docvec)| {
            let cc_size = docvec.len();
            for (doc_id, line_num) in docvec {
                annotations.entry(doc_id).or_default().insert(line_num, cc_size);
            }
            pbar.inc(1);
    });
    annotations
}

/*=================================================================
=                         WRITE OUTPUTS                           =
=================================================================*/
/*
Iterates over all paths seen, and removes the lines that we should from each
If there are no lines to kill, just copy to output
*/


fn clean_path(path: &PathBuf, lines_to_kill: Vec<usize>, output_directory: &PathBuf) -> Result<(usize, usize), Error> {
    // returns (lines_removed, lines_seen)
    let line_set: HashSet<usize> = lines_to_kill.into_iter().collect();
    let data = read_pathbuf_to_mem(path).unwrap();

    let mut output_bytes = Vec::new();
    let mut line_num = 0;
    let mut lines_removed = 0;
    for line in data.lines() {
        let line = line?;
        if line_set.contains(&line_num) {
            lines_removed += 1;
        } else {
            output_bytes.extend(line.as_bytes());
            output_bytes.push(b'\n');
        }

        line_num += 1;
    }

    if output_bytes.len() > 0 {
        let output_filename = output_directory.join(path.clone().file_name().unwrap());
        write_mem_to_pathbuf(&output_bytes, &output_filename).unwrap();
    }

    Ok((lines_removed, line_num))
}

fn scrub_all_paths(config: &MinHashConfig, lines_to_kill: DashMap<usize, DashSet<usize>>,
                   output_directory: &PathBuf) -> (usize, usize) {
    // Iterate over threads with path lookup
    let documents_seen = AtomicUsize::new(0);
    let documents_removed = AtomicUsize::new(0);
    let pbar = build_pbar(config.indices.len(), "Paths");
    config.indices.par_iter().for_each(|(path,idx)| {
        let output_filename = get_output_filename(&config.input, path, output_directory);
        let chopped_lines = lines_to_kill.get(idx).map(|v| v.value().clone());
        let (path_seen, path_removed) = chop_lines(path, &output_filename, chopped_lines).unwrap();
        documents_seen.fetch_add(path_seen, Ordering::SeqCst);
        documents_removed.fetch_add(path_removed, Ordering::SeqCst);
        pbar.inc(1);
    });

    (documents_seen.into_inner(), documents_removed.into_inner())
}


fn chop_lines(input_filename: &PathBuf, output_filename: &PathBuf, chop_lines: Option<DashSet<usize>>) -> Result<(usize, usize), Error> {
    let data = read_pathbuf_to_mem(input_filename).unwrap();
    let chop_lines: DashSet<usize> = if chop_lines.is_none() {
        DashSet::new() 
    } else {
        chop_lines.unwrap()
    };
    let mut lines_seen = 0;
    let mut lines_removed = 0;
    let mut output_bytes = Vec::new();
    let mut line_num = 0;

    for line in data.lines() {
        let line = line?;
        lines_seen += 1;
        if !chop_lines.contains(&line_num) {
            output_bytes.extend(line.as_bytes());
            output_bytes.push(b'\n');
        } else {
            lines_removed += 1;
        }
        line_num += 1;
    }

    if output_bytes.len() == 0 {
        return Ok((lines_seen, lines_removed))
    }
    write_mem_to_pathbuf(&output_bytes, output_filename).unwrap();
    Ok((lines_seen, lines_removed))

}

fn scrub_lines_survivors(config: &MinHashConfig, lines_to_survive: DashMap<usize, DashSet<usize>>, 
                         output_directory: &PathBuf) -> (usize, usize) {
    let documents_seen = AtomicUsize::new(0);
    let documents_removed = AtomicUsize::new(0);
    let pbar = build_pbar(config.indices.len(), "Paths");
    config.indices.par_iter().for_each(|(path, idx)| {
        if lines_to_survive.contains_key(idx) {
            let output_filename = get_output_filename(&config.input, path, output_directory)            ;
            let survivors = lines_to_survive.get(idx).map(|v| v.value().clone()).unwrap();
            let (path_seen, path_removed) = keep_survivors(path, &output_filename, survivors).unwrap();
            documents_seen.fetch_add(path_seen, Ordering::SeqCst);
            documents_removed.fetch_add(path_removed, Ordering::SeqCst);
        }
        pbar.inc(1);
    });
    (documents_seen.into_inner(), documents_removed.into_inner())
}

fn keep_survivors(input_filename: &PathBuf, output_filename: &PathBuf, survivors: DashSet<usize>) -> Result<(usize, usize), Error> {
    let data = read_pathbuf_to_mem(input_filename).unwrap();
    let mut lines_seen = 0;
    let mut lines_removed = 0;
    let mut output_bytes = Vec::new();
    let mut line_num = 0;
    for line in data.lines() {
        let line = line?;
        lines_seen += 1;
        if survivors.contains(&line_num) {
            output_bytes.extend(line.as_bytes());
            output_bytes.push(b'\n');            
        } else {
            lines_removed += 1;
        }
        line_num += 1;
    }
    if output_bytes.len() == 0 {
        return Ok((lines_seen, lines_removed))
    }
    write_mem_to_pathbuf(&output_bytes, output_filename).unwrap();
    Ok((lines_seen, lines_removed))
}

fn annotate_dataset(config: &MinHashConfig, annotations: DashMap<usize, DashMap<usize, usize>>, output_directory: &PathBuf) -> Result<usize, Error> {
    let documents_seen = AtomicUsize::new(0);
    let pbar = build_pbar(config.indices.len(), "Paths");
    config.indices.par_iter().for_each(|(path, idx)| {
        let binding = annotations.entry(*idx).or_default();
        let this_lookup = binding.value();
        let output_filename = get_output_filename(&config.input, path, output_directory);
        let this_docs = annotate_path(path, this_lookup, output_filename).unwrap();
        documents_seen.fetch_add(this_docs, Ordering::SeqCst);
        pbar.inc(1);
    });
    Ok(documents_seen.into_inner())
}


fn annotate_path(input_filename: &PathBuf, lookup: &DashMap<usize, usize>, output: PathBuf) -> Result<usize, Error> {
    let data = read_pathbuf_to_mem(input_filename).unwrap();
    let mut line_num = 0;
    let mut output_bytes = Vec::new();
    for line in data.lines() {
        let line = line?;
        let mut json: Value = serde_json::from_str(&line).unwrap();
        let cc_size = if lookup.contains_key(&line_num) {
            *lookup.get(&line_num).unwrap().value()
        } else {
            1
        };
        json["cc_size"] = json!(cc_size);
        let json_bytes = serde_json::to_vec(&json).unwrap();
        output_bytes.extend(json_bytes);
        output_bytes.push(b'\n');
        line_num += 1;
    }

    write_mem_to_pathbuf(&output_bytes, &output).unwrap();
    Ok(line_num)
}


/*=================================================================
=                             Subcommands                         =
=================================================================*/

fn minhash(input: &Vec<PathBuf>, output: &PathBuf, num_docs: usize, max_lines_per_path: usize, 
           sig_storage: &PathBuf, num_bands: u32, band_size: usize, ngram_size: usize) -> Result<(), Error> {
    // Note: this is only for SMALL runs. We set some hyperparameters for you, and this isn't optimized for these use cases
    return Ok(());

    /*
    let config_name = sig_storage.clone().join("config.jsonl.gz");

    build_config(input, &config_name, num_docs, max_lines_per_path).unwrap();
    hash_only(&config_name, 0, 1, num_bands, band_size, ngram_size, &vec![0 as usize], 1, 32, sig_storage, None).unwrap();
    finish_dedup(&config_name, sig_storage, output)
    */
}


fn build_config(input: &Vec<PathBuf>, output: &PathBuf, num_docs: usize, max_lines_per_path: usize,) -> Result<(), Error> {
    // Build and save the path lookup
    println!("Building config...");
    let config = MinHashConfig::new(input, num_docs, max_lines_per_path).unwrap();
    println!("Collected {:?} paths", config.len());
    let output = if has_json_extension(output) {
        output.clone() 
    } else {
        output.clone().join("config.json.gz")
    };
    config.save(&output)
}

fn build_file_map(config: &PathBuf) -> Result<(), Error> {
    /*
    - Config is the path to the json minhash config
    This function reads all the files contained in the config.input path and creates a "filemap"
    which is a .json.gz that stores the 
    map from {file_name : pathbuf -> file_id : usize}
    */
    let config_contents = read_pathbuf_to_mem(config).unwrap();
    let config_value: serde_json::Value = serde_json::from_reader(config_contents).unwrap();

    let working_dir = PathBuf::from(config_value["working_dir"].as_str().unwrap());
    create_dir_all(&working_dir).unwrap();


    let local_input = PathBuf::from(config_value["local_input"].as_str().unwrap());
    let remote_input = PathBuf::from(config_value["remote_input"].as_str().unwrap());
    let file_map = FileMap::new(&local_input, &remote_input).unwrap();

    file_map.save(&working_dir.join("filemap.json.gz"))
}


fn hash_only(config: &PathBuf, path_chunk: usize, num_path_chunks: usize) -> Result<(), Error> {
    println!("Starting part of Minhash run | config {:?} | chunk {:?}/{:?}", config, path_chunk, num_path_chunks);
    let start_main = Instant::now();    

    // Initialize everything we need to hash...
    let config_contents = read_pathbuf_to_mem(config).unwrap();
    let config_value: serde_json::Value = serde_json::from_reader(config_contents).unwrap();

    // -- Set up hashing stuff
    let perm_seed: usize = match config_value.get("x") {
        Some(val) => val.as_u64().unwrap() as usize,
        None => 0,
    };    
    let num_bands = config_value["num_bands"].as_u64().unwrap() as usize;
    let band_size = config_value["band_size"].as_u64().unwrap() as usize;
    let tokenizer_str = config_value["tokenizer_str"].as_str().unwrap();
    let ngram_size = config_value["ngram_size"].as_u64().unwrap() as usize;
    let band_seeds: Vec<u32> = _expand_band_seeds(&vec![perm_seed as u32], num_bands)
        .into_iter()
        .map(|x| x as u32)
        .collect();

    // -- Get files to hash
    let working_dir = PathBuf::from(config_value["working_dir"].as_str().unwrap());
    let file_map = FileMap::load(&PathBuf::from(working_dir.clone()).join("filemap.json.gz")).unwrap();
    let local_input = file_map.local_input.clone();    
    let this_chunk = file_map.get_path_chunk(path_chunk, num_path_chunks);    

    // -- Handle storage stuff
    let sig_storage = working_dir.clone().join("sig_storage");
    create_dir_all(&sig_storage);
    let num_sig_chunks = config_value["num_sig_chunks"].as_u64().unwrap() as usize;
    let signature_writer = SignatureWriter::new(&sig_storage, band_seeds.clone(), num_sig_chunks, path_chunk);
    let path_size = to_byte_size(file_map.indices.len());
    let line_size = to_byte_size(config_value["max_lines_per_path"].as_u64().unwrap() as usize);
    let sig_size = compute_sig_size(config_value["num_docs"].as_u64().unwrap() as usize);
    

    // And then loop through files and hash everything
    let start_hashing = Instant::now();
    let total_docs_hashed = AtomicUsize::new(0);    
    let hash_pbar = build_pbar(this_chunk.len(), "Paths");
    this_chunk.par_iter().for_each(|(path, path_id)| {
        let docs_hashed = process_path(&local_input.join(path), &band_seeds, *path_id, band_size, ngram_size, 
                                       tokenizer_str, &signature_writer, num_sig_chunks, path_size, line_size, sig_size).unwrap();
        total_docs_hashed.fetch_add(docs_hashed, Ordering::SeqCst);
        hash_pbar.inc(1);
    });
    signature_writer.finish().unwrap();
    println!("(Chunk {:?}) ...collected all hashes in {:?} seconds", path_chunk, start_hashing.elapsed().as_secs());
    println!("-------------------------");
    println!("Completing part of Minhash run | config {:?} | chunk {:?}/{:?}", config, path_chunk, num_path_chunks);
    println!("Computed hashes for {:?} bands, {:?} docs", num_bands, total_docs_hashed.into_inner());
    println!("Total runtime: {:?} (s)", start_main.elapsed().as_secs());
    Ok(())
}   



fn finish_dedup(config:&PathBuf, sig_storage: &PathBuf, output: &PathBuf) -> Result<(), Error> {
    println!("Finishing MinHash run...");
    let start_main = Instant::now();
    let config = MinHashConfig::load(config).unwrap();

    // Gather lines to kill
    let lines_to_kill = aggregate_lines_to_kill(&config, sig_storage).unwrap();


    // And then chop all the lines
    println!("Removing duplicates from pool...");
    let start_scrub = Instant::now();
    let (documents_seen, documents_removed) = scrub_all_paths(&config, lines_to_kill, &output);
    println!("... removed duplicates in {:?} seconds", start_scrub.elapsed().as_secs());


    println!("-------------------------");
    println!("Completing minhash run");
    println!("Total runtime: {:?} (s)", start_main.elapsed().as_secs());
    println!("Saw {:?} documents", documents_seen);
    println!("Removed {:?} documents", documents_removed);
    println!("Document removal rate: {:?}", documents_removed as f64 / documents_seen as f64);

    Ok(())    
}

fn gather_edges(config: &PathBuf) -> Result<(), Error> {
    println!("Starting edge gather");
    let start_main = Instant::now();

    // Load the config and initialize things
    let config_contents = read_pathbuf_to_mem(config).unwrap();
    let config_value : serde_json::Value = serde_json::from_reader(config_contents).unwrap();
    let working_dir = PathBuf::from(config_value["working_dir"].as_str().unwrap());
    let file_map = FileMap::load(&PathBuf::from(working_dir.clone()).join("filemap.json.gz")).unwrap();
    let path_size = to_byte_size(file_map.indices.len());
    let line_size = to_byte_size(config_value["max_lines_per_path"].as_u64().unwrap() as usize);
    let sig_size = compute_sig_size(config_value["num_docs"].as_u64().unwrap() as usize);

    // Gather the files into the proper groups (which should live in the same hash-space-universe)
    let edge_groups = gather_groups(working_dir.clone().join("sig_storage")).unwrap(); //(sigchunk, band_id) -> [(sigfile)]
    let single_band_id = edge_groups.iter().next().unwrap().key().1;
    let singleton_map : Option<DashMap<usize, usize>> = Some(DashMap::new()); // maps path_id -> max line_num 

    // Then build the cliques for each group of (sigchunk, band_id) -- across all files!
    let start_edges = Instant::now();

    println!("Starting edge collection...");
    let pbar = build_pbar(edge_groups.len(), "Band groups");
    edge_groups.par_iter()
        .for_each(|entry| {
            let (sigchunk, band_id) = entry.key();
            let singleton_map_opt = if *band_id == single_band_id {
                &singleton_map
            } else {
                &None
            };

            let band_group = build_band_group(entry.value(), sig_size, path_size, line_size, &singleton_map_opt).unwrap();
            let output_filename = working_dir.clone()
                                  .join("edges")
                                  .join(format!("{:08}", sigchunk))
                                  .join(format!("{:08}.edges.bin", band_id));
            save_band_group2(band_group, output_filename, path_size, line_size).unwrap();
            pbar.inc(1);
        });


    save_singletons(singleton_map.unwrap(), &working_dir.clone().join("edges").join("singletons.bin")).unwrap();
    println!("... Gathered edges in {:?} seconds", start_edges.elapsed().as_secs());
    // And save these for future use

    Ok(())
}






fn build_edges(config: &PathBuf, sig_storage: &PathBuf, group_storage: &PathBuf) -> Result<(), Error> {
    /*
    println!("Building edges...");
    let start_main = Instant::now();
    let config = MinHashConfig::load(config).unwrap();

    let start_edges = Instant::now();
    println!("Starting edge collection...");
    let band_sigs = _collect_band_sigs(sig_storage).unwrap();
    let pbar = build_pbar(band_sigs.len(), "Band Groups");
    band_sigs.par_iter()
        .for_each(|entry| {
            let (band_id, sig_chunk) = entry.key();
            let band_group = build_band_group(entry.value(), config.sig_size, config.path_size, config.line_size).unwrap();
            save_band_group(band_group, group_storage, *band_id, *sig_chunk).unwrap();
            pbar.inc(1);
        });
    println!("Completed edge building in {:?} seconds", start_edges.elapsed().as_secs());

    let start_singletons = Instant::now();
    println!("Collecting singletons...");
    let max_line_per_doc: DashMap<usize, usize> = DashMap::new(); // Maps doc_id -> max line_num
    if let Some(entry) = band_sigs.iter().next() { // TODO: Can do this with only one band sig (all chunks, only one band)
        collect_singletons(entry.value(), &max_line_per_doc, &config).unwrap();
    }
    save_singletons(max_line_per_doc, &group_storage).unwrap();
    println!("Collected singletons in {:?} seconds", start_singletons.elapsed().as_secs());


    println!("-------------------------");
    println!("Completed building all edges");
    println!("Total runtime: {:?} (s)", start_main.elapsed().as_secs());
    */
    Ok(())
}

fn build_uf2(config: &PathBuf, num_path_chunks: usize) -> Result<(), Error> {
    // Takes the edges (saved as lists of lists of (path_id, line_num) pairs)
    // and builds a union find object and then collects CC's and saves a list of ccs.
    // Unless otherwise specified, also creates a list of to-delete lines, grouped by path_id



    println!("Building UnionFind...");
    let start_main = Instant::now();

    // Load the config to initialize things
    let config_contents = read_pathbuf_to_mem(config).unwrap();
    let config_value : serde_json::Value = serde_json::from_reader(config_contents).unwrap();
    let working_dir = PathBuf::from(config_value["working_dir"].as_str().unwrap());
    let file_map = FileMap::load(&PathBuf::from(working_dir.clone()).join("filemap.json.gz")).unwrap();
    let path_size = to_byte_size(file_map.indices.len());
    let line_size = to_byte_size(config_value["max_lines_per_path"].as_u64().unwrap() as usize);

    // Build the union find and unite all the edges
    let uf = UFRush::new();
    let all_edge_files = expand_dirs(vec![working_dir.clone().join("edges")], Some(vec![".edges.bin"].as_slice())).unwrap();
    let singletons_path = working_dir.clone().join("edges").join("singletons.bin");
    let singletons = load_singletons(&singletons_path).unwrap();

    add_singletons_to_uf(singletons, &uf, line_size).unwrap();
    let pbar = build_pbar(all_edge_files.len(), "Edge files");
    all_edge_files.into_par_iter().for_each(|p| {
        add_edge_file_to_uf(&p, &uf, path_size, line_size).unwrap();
        pbar.inc(1);
    });
    println!("Built unionfind in {:?} secs", start_main.elapsed().as_secs());

    // Run the cc collection
    println!("Starting CC Collection");
    let start_cc = Instant::now();
    let ccs = build_ccs(uf, line_size);
    println!("Built ccs in {:?} secs", start_cc.elapsed().as_secs());

    // Save the list of CCs in a writer object thing
    // First push these into a dashmap and then use the MAXPATH/MAXLINE paradigm
    println!("Organizing and saving ccs...");
    let cc_map : DashMap<usize, Vec<(usize, usize)>> = DashMap::new();
    let pbar = build_pbar(ccs.len(), "CC components");
    ccs.par_iter().for_each(|((path_id, line_num), key)| {
        cc_map.entry(*key).or_default().push((*path_id, *line_num));
        pbar.inc(1);
    });

    let cc_dir = working_dir.clone().join("ccs");
    save_all_ccs(&cc_map, &cc_dir, config_value["num_sig_chunks"].as_u64().unwrap() as usize).unwrap();
        
    let kill_list = collect_kill_list(cc_map);
    let kill_dir = working_dir.clone().join("kill");
    save_kill_list(kill_list, &kill_dir, &file_map, num_path_chunks).unwrap();
    println!("Organized and saved ccs in {:?} secs", start_cc.elapsed().as_secs());


    println!("Finished all unionfind stuff in {:?} seconds" , start_main.elapsed().as_secs());
    Ok(())
}

fn uf_size_prune2(config: &PathBuf, path_chunk: usize, num_path_chunks: usize) -> Result<(), Error> {
    println!("Starting UF-based pruning...");
    let start_main = Instant::now();
    let config_contents = read_pathbuf_to_mem(config).unwrap();
    let config_value : serde_json::Value = serde_json::from_reader(config_contents).unwrap();
    let working_dir = PathBuf::from(config_value["working_dir"].as_str().unwrap());
    let output_dir = PathBuf::from(config_value["output_dir"].as_str().unwrap());
    let file_map = FileMap::load(&PathBuf::from(working_dir.clone()).join("filemap.json.gz")).unwrap();
    let path_chunk_files = file_map.get_path_chunk(path_chunk, num_path_chunks);
    let kill_dir = working_dir.clone().join("kill");

    // Parse the kill file into a map from path_id -> [lines to kill]
    println!("Reading kill file from disk...");
    let start_kill_read = Instant::now();
    let kill_file = GenWriter::get_filename(&kill_dir, path_chunk, "kill");
    let kill_list = parse_kill_file(&kill_file).unwrap();
    println!("Parsed kill file in {:?} seconds", start_kill_read.elapsed().as_secs());
    
    // Loop over the file map
    println!("Cleaning output");
    let start_kill_clean = Instant::now();
    let documents_removed = AtomicUsize::new(0);
    let documents_seen = AtomicUsize::new(0);
    let pbar = build_pbar(path_chunk_files.len(), "Files to clean");
    path_chunk_files.par_iter().for_each(|(path, path_id)| {
        let lines_to_kill = kill_list.entry(*path_id).or_default();
        let (remove, seen) = clean_path(path, lines_to_kill.to_vec(), &output_dir).unwrap();        
        documents_removed.fetch_add(remove, Ordering::SeqCst);
        documents_seen.fetch_add(seen, Ordering::SeqCst);
        pbar.inc(1);
    });
    println!("Cleaned documents in {:?} secs", start_kill_clean.elapsed());

    let documents_seen = documents_seen.into_inner();
    let documents_removed = documents_removed.into_inner();
    println!("Pruned data in {:?} secs", start_main.elapsed());
    println!("Saw {:?} lines", documents_seen);
    println!("Removed {:?} lines", documents_removed);
    println!("Removal rate was {:?}", (documents_removed as f32) / (documents_seen as f32));
    Ok(())
}


fn uf_size_prune(config: &PathBuf, ccs: &PathBuf, output: &PathBuf, floor: usize, ceil:usize) -> Result<(), Error> {
    println!("Starting UF-based pruning...");
    let start_main = Instant::now();
    let config = MinHashConfig::load(config).unwrap();

    println!("Loading CCS...");
    let start_load_cc = Instant::now();
    let loaded_ccs = load_cc_list(ccs).unwrap();
    println!("Loaded CCs in {:?} (s)", start_load_cc.elapsed().as_secs());

    println!("Grouping docs by CC...");
    let start_group_cc = Instant::now();
    let groups = group_docs_by_cc(&loaded_ccs);
    println!("Grouped CCs in {:?} (s)", start_group_cc.elapsed().as_secs());


    // Now we have to do some awkward branching:
    // If floor > 1, can just collect docs to KEEP
    println!("Starting scrub of dataset...");
    let start_scrub = Instant::now();
    let (documents_seen, documents_removed) = if floor > 0 {
        let grouped_survivors = gather_lines_to_live(groups, floor, ceil);
        scrub_lines_survivors(&config, grouped_survivors, output)
    } else {
        let grouped_ltk = gather_lines_to_kill_ccs(groups, floor, ceil);
        scrub_all_paths(&config, grouped_ltk, output)
    };
    println!("Finished scrubbing dataset in {:?} (s)", start_scrub.elapsed().as_secs());

    println!("-------------------------");
    println!("Completing minhash run");
    println!("Total runtime: {:?} (s)", start_main.elapsed().as_secs());
    println!("Saw {:?} documents", documents_seen);
    println!("Removed {:?} documents", documents_removed);
    println!("Document removal rate: {:?}", documents_removed as f64 / documents_seen as f64);

    Ok(())  
}

fn count_docs_from_sigs(config: &PathBuf, sig_storage: &PathBuf) -> Result<(), Error> {
    println!("Starting counting of docs from sigs");
    let start_main = Instant::now();
    let config = MinHashConfig::load(config).unwrap();

    let band_sigs = _collect_band_sigs(sig_storage).unwrap();
    let sig_files: Vec<PathBuf> = band_sigs.into_iter()
        .flat_map(|(_,v)| v)
        .collect();
    let counter: DashSet<(usize, usize)> = DashSet::new();
    let pbar = build_pbar(sig_files.len(), "Sig Files");
    let sig_size = config.sig_size;
    let path_size = config.path_size;
    let line_size = config.line_size;
    sig_files.into_par_iter()
        .for_each(|p| {
            let contents = read_pathbuf_to_mem(&p).unwrap().into_inner().into_inner();
            contents.chunks(sig_size + path_size + line_size)
                .for_each(|entry| {
                let path_id = IntValueEnum::from_bytes(entry[sig_size..sig_size+path_size].to_vec(), path_size).as_usize();
                let line_id = IntValueEnum::from_bytes(entry[sig_size+path_size..].to_vec(), line_size).as_usize();                
                counter.insert((path_id, line_id));
            });
            pbar.inc(1);
        });

    println!("-------------------------");
    println!("Completed counting docs from sigs");
    println!("Number of docs is {:?}", counter.len());
    println!("Total runtime: {:?} (s)", start_main.elapsed().as_secs());
    Ok(())

}

fn annotate_cc(config: &PathBuf, ccs: &PathBuf, output: &PathBuf) -> Result<(), Error> {
    // Adds a 'cc_size' attribute to all jsons and writes them to output 
    println!("Starting CC Annotation...");
    let start_main = Instant::now();
    let config = MinHashConfig::load(config).unwrap();

    println!("Loading CCS...");
    let start_load_cc = Instant::now();
    let loaded_ccs = load_cc_list(ccs).unwrap();
    println!("Loaded CCs in {:?} (s)", start_load_cc.elapsed().as_secs());

    println!("Grouping docs by CC...");
    let start_group_cc = Instant::now();
    let groups = group_docs_by_cc(&loaded_ccs);
    println!("Grouped CCs in {:?} (s)", start_group_cc.elapsed().as_secs());

    println!("Building annotation lookup...");
    let start_annotation_lookup = Instant::now();
    let annotations = build_annotation_lookup(groups);
    println!("Built annotation lookup in {:?} (s)", start_annotation_lookup.elapsed().as_secs());

    println!("Annotating dataset");
    let start_annotate = Instant::now();
    let num_docs = annotate_dataset(&config, annotations, output).unwrap();
    println!("Annotation finished in {:?} (s)", start_annotate.elapsed().as_secs());

    println!("-------------------------");
    println!("Completing CC Annotation");
    println!("Annotated {:?} docs", num_docs);
    println!("Total runtime: {:?} (s)", start_main.elapsed().as_secs());
    Ok(())    
}


/*=================================================================
=                                 MAIN                            =
=================================================================*/

fn main() {
    let args = ArgParser::parse();
    let threads = args.threads;
    if threads != 0 {
        std::env::set_var("RAYON_NUM_THREADS", threads.to_string());
    }

    let result = match &args.command {
        Commands::MinHash {input, output, num_docs, max_lines_per_path, sig_storage, num_bands, band_size, ngram_size} => {
            minhash(input, output, *num_docs, *max_lines_per_path, sig_storage, *num_bands, *band_size, *ngram_size)
        }

        Commands::BuildFileMap{config} => {
            build_file_map(config)
        }
        Commands::BuildConfig {input, output, num_docs, max_lines_per_path} => {
            build_config(input, output, *num_docs, *max_lines_per_path)
        },


        Commands::HashOnly {config, path_chunk, num_path_chunks} => {
            hash_only(config, *path_chunk, *num_path_chunks)
        },

        Commands::GatherEdges {config} => {
            gather_edges(config)
        }

        Commands::Finish {config, sig_storage, output} => {
            finish_dedup(config, sig_storage, output)        
        },

        Commands::BuildEdges {config, sig_storage, group_storage} => {
            build_edges(config, sig_storage, group_storage)
        },

        Commands::BuildUf {config, num_path_chunks} => {
            build_uf2(config, *num_path_chunks)
        },

        Commands::UfSizePrune {config, ccs, output, floor, ceil} => {
            uf_size_prune(config, ccs, output, *floor, *ceil)
        },
        Commands::AnnotateCc {config, ccs, output} => {
            annotate_cc(config, ccs, output)
        },

        Commands::CountDocsFromSigs {config, sig_storage} => {
            count_docs_from_sigs(config, sig_storage)
        },

        _ => {Ok(())}

    };
    result.unwrap()
}
