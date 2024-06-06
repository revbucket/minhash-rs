
use std::collections::{VecDeque, HashSet, HashMap};
use std::hash::{Hash, Hasher, DefaultHasher};
use anyhow::{Result, Error, anyhow};
use std::path::{PathBuf, Path};
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
use crate::io::{expand_dirs, read_pathbuf_to_mem, write_mem_to_pathbuf, get_output_filename};
use crate::storage::{to_byte_size, compute_sig_size, BandStorage, BandStorageConfig, IntValueEnum, SignatureWriter, PathLookup};
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Instant;
use std::sync::atomic::{AtomicUsize, Ordering};




pub mod s3;
pub mod io;
pub mod storage;



const MERSENNE_PRIME: u64 = (1 << 61) - 1;
const MAX_HASH: u64 = (1 << 32) - 1;

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


        #[arg(long, default_value_t=13)]
        num_bands: u32,

        #[arg(long, default_value_t=10)]
        band_size: usize,    

        #[arg(long, default_value_t=5)]
        ngram_size: usize,   

        #[arg(long, default_value_t=10_000_000_000)] // 10B docs by default?
        num_docs: usize,

        #[arg(long, default_value_t=16_000_000)] // 3 bytes by default
        max_lines_per_path: usize,




    }, 
    BuildConfigs {
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

    Hashonly {
        // Just runs and saves the hashes

        /// Location of the pre-computed path lookup object
        #[arg(required=true, long)]
        path_lookup: PathBuf,

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

        #[arg(required=true, long)] // 10B docs by default?
        storage_config_loc: PathBuf,

        #[arg(long, default_value_t=0)]
        path_chunk: usize,

        #[arg(long, default_value_t=1)]  
        num_path_chunks: usize,

        #[arg(long, default_value_t=256)]        
        num_sig_chunks: usize,

        #[arg(required=true, long)]
        sig_storage: PathBuf        
    },

    Finish {
        /// Input directories (helpful for naming output files)
        /// If left empty, this outputs according to the basename for each file
        #[arg(required=true, long)]
        input: Vec<PathBuf>,

        /// Where the path lookup lives 
        #[arg(required=true, long)]
        path_lookup: PathBuf,

        /// Where the output files go 
        #[arg(required=true, long)]
        output: PathBuf,        

        /// where the StorageConfig lives
        #[arg(required=true, long)]
        sig_storage: PathBuf,

        /// where the StorageConfig lives
        #[arg(required=true, long)]
        storage_config_loc: PathBuf,
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
                config: &BandStorageConfig, signature_writer: &SignatureWriter, num_sig_chunks: usize) -> Result<(), Error> {
    // Setup things: load data, build tokenizer, etc
    let data = read_pathbuf_to_mem(path).unwrap();
    // let mut buffer = Vec::new();
    // data.read_to_end(&mut buffer).unwrap();
    // println!("READ DATA {:?}", buffer);
    let tokenizer = p50k_base().unwrap();
    let num_bands = band_seeds.len();
    let perm_seeds = _expand_band_seeds(&band_seeds, band_size);
    let path_id = IntValueEnum::new(path_id, config.path_size);
    for (line_num, line) in data.lines().enumerate() {

        let line_num = IntValueEnum::new(line_num, config.line_size);
        let line = line.unwrap();
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
    Ok(())
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

fn _save_band_signature(band_storage: &BandStorage, band_seed: u64, band_signature: IntValueEnum, doc_id: (IntValueEnum, IntValueEnum)) -> () {
    // TODO: ADD BAND SIGNATURE
    band_storage
        .entry(band_seed)
        .or_insert_with(|| DashMap::new())
        .entry(band_signature)
        .or_default()
        .push(doc_id);
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

fn aggregate_lines_to_kill(storage_config_loc: &PathBuf, sig_storage: &PathBuf) -> 
    Result<DashMap<usize, DashSet<usize>>, Error> {
    println!("Aggregating hashes into lines to kill...");
    let start_main = Instant::now();
    let storage_config = BandStorageConfig::load(storage_config_loc).unwrap();

    // Gather all files in storage and group by (band, sigchunk)
    let stored_files : DashMap<(u32, usize), Vec<PathBuf>> = DashMap::new();
    let all_files = expand_dirs(vec![sig_storage.clone()], Some(vec![".bin"].as_slice())).unwrap();
    all_files.par_iter()
        .for_each(|p| {
            let key = _extract_bandid_sigchunk(p).unwrap();
            stored_files.entry(key).or_default().push(p.clone());
        });


    let lines_to_kill : DashMap<usize, DashSet<usize>> = DashMap::new();
    let lines_to_kill_pbar = build_pbar(stored_files.len(), "File groups");
    stored_files.par_iter()
        .for_each(|v| {
            _augment_lines_to_kill(&lines_to_kill, &v, storage_config.sig_size, storage_config.path_size, storage_config.line_size).unwrap();
            lines_to_kill_pbar.inc(1);
        });

    println!("-------------------------");
    println!("Aggregated all lines to kill in {:?}", start_main.elapsed().as_secs());
    Ok(lines_to_kill)
}

    
fn _extract_bandid_sigchunk(path: &PathBuf) -> Result<(u32, usize), Error> {
    let file_name = path.file_name().unwrap().to_str().unwrap();
    let re = Regex::new(r"band_(\d+)/sigchunk(\d+)_filechunk\d+\.bin").unwrap();
    
    if let Some(captures) = re.captures(file_name) {
        let band = captures[1].parse::<u32>().ok().unwrap();
        let sigchunk = captures[2].parse::<usize>().ok().unwrap();
        Ok((band, sigchunk))
    } else {        
        Err(anyhow!("Failed to extract band_id/sig_chunk"))
    }
}


fn _augment_lines_to_kill(lines_to_kill: &DashMap<usize, DashSet<usize>>, paths: &Vec<PathBuf>, 
                          sig_size: usize, path_size: usize, line_size: usize) -> Result<(), Error> {

    // Load all data and create mapping of {sig -> [(doc_id, line_num),...]}
    let entry_size = sig_size + path_size + line_size;
    let mut groups : HashMap<IntValueEnum, Vec<(IntValueEnum, IntValueEnum)>> = HashMap::new();
    for path in paths {
        let contents = read_pathbuf_to_mem(path).unwrap().into_inner().into_inner();
        for entry in contents.chunks(entry_size) {
            let sig = IntValueEnum::from_bytes(entry[..sig_size].to_vec(), sig_size);
            let path_id = IntValueEnum::from_bytes(entry[sig_size..sig_size+path_size].to_vec(), path_size);
            let line_size = IntValueEnum::from_bytes(entry[sig_size+path_size..].to_vec(), line_size);
            groups.entry(sig).or_default().push((path_id, line_size));
        }
    }
    // Q: Maybe sorting is faster here???

    // Then add all but the first (path, line) to the lines_to_kill list
    for value in groups.values_mut() {
        while value.len() > 1 {
            let entry = value.pop().unwrap();
            lines_to_kill.entry(entry.0.as_usize()).or_default().insert(entry.1.as_usize());
        }
    }
    Ok(())
}



/*=================================================================
=                         WRITE OUTPUTS                           =
=================================================================*/
/*
Iterates over all paths seen, and removes the lines that we should from each
If there are no lines to kill, just copy to output
*/
fn scrub_all_paths(path_lookup: &PathLookup, lines_to_kill: DashMap<usize, DashSet<usize>>,
                   input_dirs: &Vec<PathBuf>, output_directory: &PathBuf) -> (usize, usize) {
    // Iterate over threads with path lookup
    let documents_seen = AtomicUsize::new(0);
    let documents_removed = AtomicUsize::new(0);
    let pbar = build_pbar(path_lookup.indices.len(), "Paths");
    path_lookup.indices.par_iter().for_each(|entry| {
        let input_filename = entry.key();
        let path_id = entry.value();
        let output_filename = get_output_filename(input_dirs, input_filename, output_directory);
        let chopped_lines = lines_to_kill.get(path_id).map(|v| v.value().clone());
        let (path_seen, path_removed) = chop_lines(input_filename, &output_filename, chopped_lines).unwrap();
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

/*
fn minhash(input: &Vec<PathBuf>, output: &PathBuf, num_bands: u32, band_size: usize, ngram_size: usize, num_docs: usize, max_lines_per_path: usize) -> Result<(), Error>{
    println!("Starting MinHash run...");    
    let start_main = Instant::now();
    // Phase 0: Setup, collect filenames, build path lookup, build band seeds
    let mut input_files = expand_dirs(input.clone(), None).unwrap();
    input_files.sort(); // sort before building the path lookup
    println!("Collected {:?} input files", input_files.len());
    let path_lookup = PathLookup::new(input_files.clone());
    let band_seeds: Vec<u32> = (0..num_bands).map(|i| i as u32).collect();


    let sig_size = compute_sig_size(num_docs);
    let path_size = to_byte_size(path_lookup.len());
    let line_size = to_byte_size(max_lines_per_path);
    println!("SIZES {:?} {:?} {:?}", sig_size, path_size, line_size);
    let band_storage_config = BandStorageConfig { sig_size, path_size, line_size };

    // Phase 1: Collect hashes for everything
    println!("Starting hash collection...");
    let start_hashing = Instant::now();

    let band_storage = BandStorage::new();
    let hash_pbar = build_pbar(input_files.len(), "Paths");
    path_lookup.indices.par_iter().for_each(|entry| {
        let input_filename = entry.key();
        let path_id = entry.value();
        process_path(input_filename, &band_seeds, *path_id, band_size, ngram_size, &band_storage, &band_storage_config).unwrap();
        hash_pbar.inc(1) // Claude promises me this is threadsafe
    });
    //let band_storage_size = est_storage_size_in_bytes(&band_storage);
    println!("...collected all hashes in {:?} seconds", start_hashing.elapsed().as_secs());

    // Phase 2: Build Build lines to kill 
    println!("Collecting which documents to remove...");
    let start_line_to_kill = Instant::now();
    let lines_to_kill = build_lines_to_kill(&band_storage);
    println!("...collected which lines to kill in {:?} seconds", start_line_to_kill.elapsed().as_secs());
    save_lines_to_kill(lines_to_kill.clone(), &Path::join(&output, "lines_to_kill.gz")).unwrap();

    // Phase 3: Chop all the lines
    println!("Removing duplicates from pool...");
    let start_scrub = Instant::now();
    let (documents_seen, documents_removed) = scrub_all_paths(&path_lookup, lines_to_kill, &input, &output);
    println!("... removed duplicates in {:?} seconds", start_scrub.elapsed().as_secs());


    // Final report
    println!("-------------------------");
    println!("Completing minhash run");
    println!("Total runtime: {:?} (s)", start_main.elapsed().as_secs());
    //println!("Band storage size {}", human_bytes(band_storage_size as f64));
    println!("Saw {:?} documents", documents_seen);
    println!("Removed {:?} documents", documents_removed);
    println!("Document removal rate: {:?}", documents_removed as f64 / documents_seen as f64);

    Ok(())
}
*/


fn build_configs(input: &Vec<PathBuf>, output: &PathBuf, num_docs: usize, max_lines_per_path: usize) -> Result<(), Error> {
    // Build and save the path lookup
    let mut input_files = expand_dirs(input.clone(), None).unwrap();
    input_files.sort(); // sort before building the path lookup
    println!("Collected {:?} input files", input_files.len());
    let path_lookup = PathLookup::new(input_files.clone());
    let path_lookup_filename = output.clone().join("path_lookup.gz");
    path_lookup.save_to_file(&path_lookup_filename).unwrap();

    // Build and save the storage config 
    let storage_config_filename = output.clone().join("storage_config.json");
    let storage_config = BandStorageConfig::infer_new(num_docs, input_files.len(), max_lines_per_path);
    storage_config.save(storage_config_filename).unwrap();

    Ok(())
}


fn hash_only(path_lookup: &PathBuf, band_group_id: usize, band_start: u32, num_bands: u32, 
              band_size: usize, ngram_size: usize, storage_config_loc: &PathBuf,
              path_chunk: usize, num_path_chunks: usize, num_sig_chunks: usize, sig_storage: &PathBuf) -> Result<(), Error> {
    println!("Starting part of MinHash run (band group {:?})...", band_group_id);        
    let start_main = Instant::now();
    
    // Load path lookup and setup things    
    let path_lookup = PathLookup::load_from_file(path_lookup).unwrap();
    let band_seeds: Vec<u32> = if band_start == 0 {
        let mut rng = rand::thread_rng();
        (0..num_bands).map(|_| rng.gen()).collect()
    } else {
        (band_start..band_start+num_bands).collect()
    };
    let band_storage_config = BandStorageConfig::load(storage_config_loc).unwrap();
    let signature_writer = SignatureWriter::new(sig_storage, band_seeds.clone(), num_sig_chunks, path_chunk);
    let path_chunk = path_lookup.get_chunk(path_chunk, num_path_chunks);

    println!("Starting hash collection...");
    let start_hashing = Instant::now();
    let hash_pbar = build_pbar(path_lookup.len(), "Paths");

    path_chunk.par_iter().for_each(|(path, path_id)| {
        process_path(path, &band_seeds, *path_id, band_size, ngram_size, &band_storage_config,
                     &signature_writer, num_sig_chunks).unwrap();
        hash_pbar.inc(1) // Claude promises me this is threadsafe with rayon
    });
    println!("...collected all hashes in {:?} seconds", start_hashing.elapsed().as_secs());

    let save_to_s3 = false; 
    if save_to_s3 {
        ()
    }
    println!("-------------------------");
    println!("Completing part of MinHash run (band group {:?} | )", band_group_id);
    println!("Total runtime: {:?} (s)", start_main.elapsed().as_secs());
    return Ok(());
}


fn finish_dedup(input: &Vec<PathBuf>, path_lookup: &PathBuf, output: &PathBuf, sig_storage: &PathBuf, 
                storage_config_loc: &PathBuf) -> Result<(), Error> {
    println!("Finishing MinHash run...");
    let start_main = Instant::now();
    let path_lookup = PathLookup::load_from_file(path_lookup).unwrap();


    // Gather lines to kill
    let lines_to_kill = aggregate_lines_to_kill(storage_config_loc, sig_storage).unwrap();


    // And then chop all the lines
    println!("Removing duplicates from pool...");
    let start_scrub = Instant::now();
    let (documents_seen, documents_removed) = scrub_all_paths(&path_lookup, lines_to_kill, &input, &output);
    println!("... removed duplicates in {:?} seconds", start_scrub.elapsed().as_secs());


    println!("-------------------------");
    println!("Completing minhash run");
    println!("Total runtime: {:?} (s)", start_main.elapsed().as_secs());
    println!("Saw {:?} documents", documents_seen);
    println!("Removed {:?} documents", documents_removed);
    println!("Document removal rate: {:?}", documents_removed as f64 / documents_seen as f64);

    Ok(())    
}


/*=================================================================
=                                 MAIN                            =
=================================================================*/

fn main() {
    let args = ArgParser::parse();

    let result = match &args.command {
        Commands::BuildConfigs {input, output, num_docs, max_lines_per_path} => {
            build_configs(input, output, *num_docs, *max_lines_per_path)
        },
        Commands::Hashonly {path_lookup, band_group_id, band_start, num_bands, band_size, ngram_size, storage_config_loc,
                             path_chunk, num_path_chunks, num_sig_chunks, sig_storage} => {
            hash_only(path_lookup, *band_group_id, *band_start, *num_bands, *band_size, *ngram_size, storage_config_loc,
                       *path_chunk, *num_path_chunks, *num_sig_chunks, sig_storage)
        },

        Commands::Finish {input, path_lookup, output, sig_storage, storage_config_loc,} => {
            finish_dedup(input, path_lookup, output, sig_storage, storage_config_loc)        
        }, 
        _ => { Ok(()) }

    };
    result.unwrap()
}
