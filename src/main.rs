
use serde::{Serialize, Deserialize};
use std::collections::{VecDeque, HashMap};
use std::hash::{Hash, Hasher, DefaultHasher};
use anyhow::{Result, Error, anyhow};
use std::path::{PathBuf};
use std::io::{BufRead};
use rand::prelude::*;
use tiktoken_rs::{p50k_base, CoreBPE};
use serde_json::Value;
use regex::Regex;
use ndarray::{Array1};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sha2::{Sha256, Digest};
use dashmap::{DashMap, DashSet};
use rayon::prelude::*;
use clap::{Parser, Subcommand};
use crate::io::{expand_dirs, read_pathbuf_to_mem, write_mem_to_pathbuf, get_output_filename, has_json_extension};
use crate::storage::{IntValueEnum, SignatureWriter, MinHashConfig};
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Instant;
use std::sync::atomic::{AtomicUsize, Ordering};
use crate::uf::{UnionFind};
use bincode;

pub mod uf;
pub mod s3;
pub mod io;
pub mod storage;



const MERSENNE_PRIME: u64 = (1 << 61) - 1;
const MAX_HASH: u64 = (1 << 32) - 1;
const CC_CHUNK_SIZE: usize = 100 * 1024 * 1024; // 100 MB chunk size


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


        #[arg(long, default_value_t=13)]
        num_bands: u32,

        #[arg(long, default_value_t=10)]
        band_size: usize,    

        #[arg(long, default_value_t=5)]
        ngram_size: usize,   
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

        /// Location of the pre-computed path lookup object
        #[arg(required=true, long)]
        config: PathBuf,

        /// Give a unique id for this run 
        #[arg(required=true, long)]
        band_group_id: usize,

        /// Band start (needed for full determinism. Leaving this unset is probably okay)
        #[arg(long, default_value_t=0)] // 0 is default
        band_start: u32,

        #[arg(long, default_value_t=13)]
        num_bands: u32,

        #[arg(long, default_value_t=10)]
        band_size: usize,    

        #[arg(long, default_value_t=5)]
        ngram_size: usize,   

        #[arg(long, num_args=1.., default_value ="0,")]
        path_chunk: Vec<usize>,

        #[arg(long, default_value_t=1)]  
        num_path_chunks: usize,

        #[arg(long, default_value_t=256)]        
        num_sig_chunks: usize,

        #[arg(required=true, long)]
        sig_storage: PathBuf,

        #[arg(long)]     
        s3_storage: Option<PathBuf>
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
        group_storage: PathBuf,

        #[arg(required=true, long)]
        output: PathBuf
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
                config: &MinHashConfig, signature_writer: &SignatureWriter, num_sig_chunks: usize) -> Result<usize, Error> {
    // Setup things: load data, build tokenizer, etc
    let data = read_pathbuf_to_mem(path).unwrap();
    // let mut buffer = Vec::new();
    // data.read_to_end(&mut buffer).unwrap();
    // println!("READ DATA {:?}", buffer);
    let tokenizer = p50k_base().unwrap();
    let num_bands = band_seeds.len();
    let perm_seeds = _expand_band_seeds(&band_seeds, band_size);
    let path_id = IntValueEnum::new(path_id, config.path_size);
    let mut docs_hashed = 0;
    for (line_num, line) in data.lines().enumerate() {

        let line_num = IntValueEnum::new(line_num, config.line_size);
        let line = line.unwrap();
        docs_hashed += 1;
        let json: Value = serde_json::from_str(&line).unwrap();
        let text = json["text"].as_str().unwrap();
        let tokens = preprocess_text(text, &tokenizer);
        let hash_vals = get_hash_vals_from_tokens(tokens, &perm_seeds, ngram_size);
        let bands = hash_vals.into_shape((num_bands, band_size)).unwrap();
        for (row, band_seed) in bands.rows().into_iter().zip(band_seeds.iter()) {
            let mut hasher = Sha256::new(); 
            hasher.update(bytemuck::cast_slice(row.as_slice().unwrap()));
            let hash = hasher.finalize();
            let band_signature = IntValueEnum::from_bytes(hash[..config.sig_size].to_vec(), config.sig_size);   
            _save_band_signature_to_disk(&signature_writer, *band_seed, band_signature, path_id.clone(), line_num.clone(), num_sig_chunks).unwrap();
            //_save_band_signature(band_storage, *band_seed, band_signature, doc_id.clone());
        }
    }
    Ok(docs_hashed)
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
    phv.zip_mut_with(&b, |x, y| *x = ((x.wrapping_mul(cur_hash).wrapping_add(*y)) % MERSENNE_PRIME) & MAX_HASH);
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
    let all_files = expand_dirs(vec![sig_storage.clone()], Some(vec![".bin"].as_slice())).unwrap();
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


fn build_band_group(band_sigs: Vec<PathBuf>, sig_size: usize, path_size: usize, line_size: usize) -> 
    Result<Vec<Vec<(usize, usize)>>, Error> {
    // For a group of files that contain signatures within the same band (and a sig chunk)
    // Collects a list of (path_id: usize, line_id: usize) for each clique
    // (reading each file is done in parallel, so nothing upstream should be par_iter'ed)

    let entry_size = sig_size + path_size + line_size;
    let group_map : DashMap<IntValueEnum, Vec<(usize, usize)>> = DashMap::new();


    band_sigs.iter().for_each(|path| {
        let contents = read_pathbuf_to_mem(path).unwrap().into_inner().into_inner();
        contents.par_chunks(entry_size).for_each(|entry| {
            let sig = IntValueEnum::from_bytes(entry[..sig_size].to_vec(), sig_size);
            let path_id = IntValueEnum::from_bytes(entry[sig_size..sig_size+path_size].to_vec(), path_size).as_usize();
            let line_id = IntValueEnum::from_bytes(entry[sig_size+path_size..].to_vec(), line_size).as_usize();
            group_map.entry(sig).or_default().push((path_id, line_id));
        });
    });
    
    let band_group: Vec<Vec<(usize, usize)>> = group_map
        .into_par_iter()
        .map(|(_, group)| group)
        .filter(|value| value.len() >1)
        .collect();

    Ok(band_group)
}

fn save_band_group(band_group: Vec<Vec<(usize, usize)>>, output: &PathBuf, band_id: u32, sig_chunk: usize) 
    -> Result<(), Error> {
    let band_group_name = _get_band_group_name(output, band_id, sig_chunk);
    let serialized: Vec<u8> = bincode::serialize(&BandGroup(band_group)).unwrap();
    write_mem_to_pathbuf(&serialized, &band_group_name)
}


fn _get_band_group_name(band_group_storage: &PathBuf, band_id: u32, sig_chunk: usize) -> PathBuf {
    band_group_storage.clone()
        .join(format!("band_{:016}", band_id))
        .join(format!("bandgroup{:08}.group.bin", sig_chunk))
}

fn add_band_group_to_uf(band_group: &PathBuf, uf: UnionFind) -> Result<UnionFind, Error> {
    // Adds all groups in the band group to the unionfind

    let serialized = read_pathbuf_to_mem(band_group).unwrap().into_inner().into_inner();
    let band_group: Vec<Vec<(usize, usize)>> = bincode::deserialize(&serialized).unwrap();
    
    band_group
        .into_par_iter()
        .for_each(|mut group| {
            let last = group.pop().unwrap();
            while group.len() > 0 {
                //uf.union(last, group.pop().unwrap());
            }
    });
    Ok(uf)
}

/*=================================================================
=                         WRITE OUTPUTS                           =
=================================================================*/
/*
Iterates over all paths seen, and removes the lines that we should from each
If there are no lines to kill, just copy to output
*/
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




/*=================================================================
=                             Subcommands                         =
=================================================================*/

fn minhash(input: &Vec<PathBuf>, output: &PathBuf, num_docs: usize, max_lines_per_path: usize, 
           sig_storage: &PathBuf, num_bands: u32, band_size: usize, ngram_size: usize) -> Result<(), Error> {
    // Note: this is only for SMALL runs. We set some hyperparameters for you, and this isn't optimized for these use cases
    let config_name = sig_storage.clone().join("config.jsonl.gz");

    build_config(input, &config_name, num_docs, max_lines_per_path).unwrap();
    hash_only(&config_name, 0, 1, num_bands, band_size, ngram_size, &vec![0 as usize], 1, 32, sig_storage, None).unwrap();
    finish_dedup(&config_name, sig_storage, output)
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


fn hash_only(config: &PathBuf, band_group_id: usize, band_start: u32, num_bands: u32, 
              band_size: usize, ngram_size: usize, 
              path_chunk: &Vec<usize>, num_path_chunks: usize, num_sig_chunks: usize, sig_storage: &PathBuf,
              s3_storage: Option<&PathBuf>) -> Result<(), Error> {
    println!("Starting part of MinHash run (band group {:?})...", band_group_id);        
    let start_main = Instant::now();    

    // Load config and setup things      
    let config = MinHashConfig::load(config).unwrap();
    let band_seeds: Vec<u32> = if band_start == 0 {
        let mut rng = rand::thread_rng();
        (0..num_bands).map(|_| rng.gen()).collect()
    } else {
        (band_start..band_start+num_bands).collect()
    };
    let total_docs_hashed = AtomicUsize::new(0);    
    for cur_chunk in path_chunk {
        let cur_chunk = *cur_chunk;
        let signature_writer = SignatureWriter::new(sig_storage, band_seeds.clone(), num_sig_chunks, cur_chunk);
        let chunked_paths = config.get_chunk(cur_chunk, num_path_chunks);

        // Collect hashes to disk
        println!("(Chunk {:?}) Starting hash collection... ", cur_chunk);
        let start_hashing = Instant::now();
        let hash_pbar = build_pbar(chunked_paths.len(), "Paths");
        chunked_paths.par_iter().for_each(|(path, path_id)| {
            let docs_hashed = process_path(path, &band_seeds, *path_id, band_size, ngram_size, &config,
                         &signature_writer, num_sig_chunks).unwrap();
            total_docs_hashed.fetch_add(docs_hashed, Ordering::SeqCst);
            hash_pbar.inc(1) // Claude promises me this is threadsafe with rayon
        });
        signature_writer.finish().unwrap();
        println!("(Chunk {:?}) ...collected all hashes in {:?} seconds", cur_chunk, start_hashing.elapsed().as_secs());

        // Save to s3 if specified
        if !s3_storage.is_none() {
            let s3_storage = s3_storage.unwrap();
            let io_pairs = signature_writer.get_input_output_filenames(&s3_storage, cur_chunk);
            io_pairs.par_iter()
                .for_each(|(input_path, output_path)| {
                    let data = read_pathbuf_to_mem(&input_path).unwrap();
                    let data = data.into_inner().into_inner();
                    let data = data.as_slice();
                    write_mem_to_pathbuf(&data, &output_path).unwrap();
            });
        }
    }

    // Summarize outputs    
    println!("-------------------------");
    println!("Completing part of MinHash run (band group {:?} | path chunk(s) {:?})", band_group_id, path_chunk);
    println!("Computed hashes for {:?} bands, {:?} docs", num_bands, total_docs_hashed.into_inner());
    println!("Total runtime: {:?} (s)", start_main.elapsed().as_secs());
    return Ok(());
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

fn build_edges(config: &PathBuf, sig_storage: &PathBuf, group_storage: &PathBuf) -> Result<(), Error> {
    println!("Building edges...");
    let start_main = Instant::now();
    let config = MinHashConfig::load(config).unwrap();

    let band_sigs = _collect_band_sigs(sig_storage).unwrap();

    let pbar = build_pbar(band_sigs.len(), "Band Groups");
    band_sigs.into_iter()
        .for_each(|(k, v)| {
            let (band_id, sig_chunk) = k;
            let band_group = build_band_group(v, config.sig_size, config.path_size, config.line_size).unwrap();
            save_band_group(band_group, group_storage, band_id, sig_chunk).unwrap();
            pbar.inc(1);
        });
    println!("-------------------------");
    println!("Completed building all edges");
    println!("Total runtime: {:?} (s)", start_main.elapsed().as_secs());
    Ok(())
}

fn build_uf(band_group_storage: &PathBuf, output: &PathBuf) -> Result<(), Error> {
    // Saves a {connected-component: [(doc_id, line_id)]} dict built from the band groups

    println!("Building UnionFind...");
    let start_main = Instant::now();

    let mut uf = UnionFind::new();
    let all_band_groups = expand_dirs(vec![band_group_storage.clone()], Some(vec![".sav"].as_slice())).unwrap();
    all_band_groups.into_iter().for_each(|band_group| {
        //uf = add_band_group_to_uf(&band_group, uf).unwrap();
    });
    println!("Built union find in {:?} secs", start_main.elapsed().as_secs());

    let ccs = uf.get_ccs();

    // Save the cc chunk (do this as a method in the UF?)
    let serialized_ccs = bincode::serialize(&ccs).unwrap();

    serialized_ccs.par_chunks(CC_CHUNK_SIZE)
        .enumerate()        
        .for_each(|(idx, chunk)| {
            let part_name = output.clone().join(format!("cc_part{:08}.cc.bin", idx));
            write_mem_to_pathbuf(chunk, &part_name).unwrap();
    });

    println!("-------------------------");
    println!("Completed building UnionFind");
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

        Commands::BuildConfig {input, output, num_docs, max_lines_per_path} => {
            build_config(input, output, *num_docs, *max_lines_per_path)
        },
        Commands::HashOnly {config, band_group_id, band_start, num_bands, band_size, ngram_size, 
                             path_chunk, num_path_chunks, num_sig_chunks, sig_storage, s3_storage} => {
            hash_only(config, *band_group_id, *band_start, *num_bands, *band_size, *ngram_size,
                       path_chunk, *num_path_chunks, *num_sig_chunks, sig_storage, s3_storage.as_ref())
        },

        Commands::Finish {config, sig_storage, output} => {
            finish_dedup(config, sig_storage, output)        
        },

        Commands::BuildEdges {config, sig_storage, group_storage} => {
            build_edges(config, sig_storage, group_storage)
        }

        Commands::BuildUf {group_storage, output} => {
            build_uf(group_storage, output)
        }

        _ => {Ok(())}

    };
    result.unwrap()
}
