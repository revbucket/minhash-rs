
use std::collections::{VecDeque, HashSet, HashMap};
use std::hash::{Hash, Hasher, DefaultHasher};
use anyhow::{Result, Error};
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
use dashmap::DashMap;
use rayon::prelude::*;
use clap::{Parser};
use crate::io::{expand_dirs, read_pathbuf_to_mem, write_mem_to_pathbuf, get_output_filename};
use crate::storage::{to_byte_size, compute_sig_size, BandStorage, BandStorageConfig, IntValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Instant;
use std::sync::atomic::{AtomicUsize, Ordering};
use human_bytes::human_bytes;




pub mod s3;
pub mod io;
pub mod storage;



const MERSENNE_PRIME: u64 = (1 << 61) - 1;
const MAX_HASH: u64 = (1 << 32) - 1;


//                           band.         signature       pathID LineNum
/*
General design notes for minhash in rust:
Barebones/not too many features

V1: Run the whole shebang just to get the steps right
    1. Collect a bunch of files and build/save a struct that lets you 
        a. put in a filename (whole s3 uri, file path) -> get out file index (usize)
        b. put in a file index (usize) -> get out a usize index
    2. For each file, build the hashes and store them in a shared data structure
        a. For each document in each file:
            i.    preprocess the text
            ii.   tokenize it (tiktoken)
            iii.  create ngram shinglings and hash each one 
            iv.   for each signature (vectorized):
                  take mins over all ngrams and make the signature value
            v.    group across bands and hash each band
            vi.   save (path_id, doc_id, band, band-signature) in some threadshared datastructure
        
    3. group all (path_id, doc_id) by (band, band-signature), put these in a list
    4. Keep collection of (path_id, doc_id) to kill in shared set 
        (will kill unless is first in list. Will collect all minimal elements in each band clique.
         In the larger graph, if assuming directedness by some ordering, the only things kept are
         the minimal elements
        )
    5. Reprocess all elements and kill according to rules. Don't write empty docs, etc etc


V2: 
    1. Have command to save step2 outputs somewhere 
    2. Load all step2 outputs into memory
    3. Do step 3,4,5 as above


NOTE:
    general pattern for async-y things is that we will init a runtime 
    to interact with s3 (read, write ,list files) and block until that's done. 
*/



/*=================================================================
=                                  ARGS                           =
=================================================================*/

#[derive(Parser, Debug)]
struct Args {
    /// (List of) directories/files (on s3 or local) that are jsonl.gz or jsonl.zstd files
    #[arg(required=true, long, num_args=1..)]
    input: Vec<PathBuf>,

    /// Output location (may be an s3 uri)
    #[arg(required=true, long)]
    output: PathBuf,


    #[arg(long, default_value_t=13)]
    num_bands: usize,

    #[arg(long, default_value_t=10)]
    band_size: usize,    

    #[arg(long, default_value_t=5)]
    ngram_size: usize,   

    #[arg(long, default_value_t=10_000_000_000)] // 10B docs by default?
    num_docs: usize,

    #[arg(long, default_value_t=16_000_000)] // 3 bytes by default
    max_lines_per_path: usize

}

/*=================================================================
=                             UTILITIES                           =
=================================================================*/


fn build_pbar(num_items: usize, units: &str) -> ProgressBar {
    let mut template = String::from(units);
    template.push_str(" {human_pos}/{human_len} [{elapsed_precise}/{duration_precise}] [{wide_bar:.cyan/blue}]");
    ProgressBar::new(num_items as u64)
        .with_style(
            ProgressStyle::with_template(&template).unwrap()
        )
}


fn est_storage_size_in_bytes(band_storage: &BandStorage) -> usize {
    // Quick'n'dirty way to estimate how much RAM we need for the hash object 
    // Calculate like: 
    // |Outer dashmap keys| * 8
    // For each inner dashmap:
    //    + |Inner dashmap keys| * 16
    //    + |total_vals| * 16
    panic!("NOT IMPLEMENTED");
    let usize_size = std::mem::size_of::<usize>();
    let size = AtomicUsize::new(0);
    size.fetch_add(band_storage.len() * 8, Ordering::SeqCst); // Keys

    for band_ref in band_storage.iter() {
        size.fetch_add(band_ref.value().len() * 16, Ordering::SeqCst); // 16 for each key in inner
        band_ref.value().par_iter().for_each(|entry| {
            size.fetch_add(entry.value().len() * usize_size, Ordering::SeqCst);
        });
    }

    size.into_inner()
}





/*=================================================================
=                               PATH LOOKUP                       =
=================================================================*/
/*
Struct that saves the order of files. Useful for making sure we have the same order.
Basically just used to map filenames to usizes and vice versa so we only have to
pass around indices vs whole strings
*/
struct PathLookup {
    //paths: Vec<PathBuf>,
    indices: DashMap<PathBuf, usize>,
}

impl PathLookup {
    fn new(paths: Vec<PathBuf>) -> Self {
        let indices = Self::build_indices(&paths);
        PathLookup {// paths, 
                    indices }
    }

    fn build_indices(paths: &Vec<PathBuf>) -> DashMap<PathBuf, usize> {
        paths
            .iter()
            .enumerate()
            .map(|(index, path)| (path.clone(), index))
            .collect()
    }

    fn len(&self) -> usize {
        self.indices.len()
    }

    fn save_to_file(&self, file_path: &PathBuf) -> Result<(), Error> {
        let hash_indices: HashMap<_, _> = self.indices.clone().into_par_iter().collect();
        let serialized = serde_json::to_vec(&hash_indices)?;
        write_mem_to_pathbuf(&serialized, file_path)
    }

    fn load_from_file(file_path: &PathBuf) -> Result<Self> {
        let contents = read_pathbuf_to_mem(file_path).unwrap();
        let deserialized: HashMap<PathBuf, usize> = serde_json::from_reader(contents)?;
        let mut pairs: Vec<(PathBuf, usize)> = deserialized.into_iter().collect();
        pairs.sort_by_key(|&(_, id)| id);
        let paths = pairs.into_iter().map(|(path, _)| path).collect();
        Ok(PathLookup::new(paths))
    }

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


fn process_path(path: &PathBuf, band_seeds: &Vec<u64>, path_id: usize, band_size: usize, ngram_size: usize,
                band_storage: &BandStorage, config: &BandStorageConfig) -> Result<(), Error> {
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
        let doc_id = (path_id.clone(), IntValueEnum::new(line_num, config.line_size));
        let line = line.unwrap();
        let json: Value = serde_json::from_str(&line).unwrap();
        let text = json["text"].as_str().unwrap();

        let tokens = preprocess_text(text, &tokenizer);
        let hash_vals = get_hash_vals_from_tokens(tokens, &perm_seeds, ngram_size);
        let bands = hash_vals.into_shape((num_bands, band_size)).unwrap();
        for (row, band_seed) in bands.rows().into_iter().zip(band_seeds.iter()) {
            // hash each band signature to 128 bits to minimize collisions
            let mut hasher = Sha256::new(); 
            hasher.update(bytemuck::cast_slice(row.as_slice().unwrap()));
            let hash = hasher.finalize();
            //println!("CONFIG SIG SIZE {:?} ", config.sig_size);
            //println!("SIG {:?}", hash[..config.sig_size].to_vec());
            //let band_signature = IntValueEnum::from_bytes(hash[..config.sig_size].to_vec(), config.sig_size);   

            //println!("GOT BAND SIG");

            let band_signature = IntValueEnum::from_bytes(hash[..config.sig_size].to_vec(), config.sig_size);   
            _save_band_signature(band_storage, *band_seed, band_signature, doc_id.clone());
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

fn _expand_band_seeds(band_seeds: &Vec<u64>, band_size: usize) -> Vec<u64> {
    // Each "band seed" is expanded here to band_size random u64s, and flattened. (used to seed permutations)
    // Probably like no collisions here, so let's just not worry about that ;) 

    let mut perm_seeds: Vec<u64> = Vec::new();
    for band_seed in band_seeds.iter() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(*band_seed);
        for _i in 0..band_size {
            perm_seeds.push(rng.next_u64());
        }
    }
    perm_seeds
}

fn _save_band_signature(band_storage: &BandStorage, band_seed: u64, band_signature: IntValueEnum, doc_id: (IntValueEnum, IntValueEnum)) -> () {
    // TODO: ADD BAND SIGNATURE
    band_storage.entry(band_seed).or_default()
                  .entry(band_signature).or_default().push(doc_id);
}

/*=================================================================
=                      COLLECT DOCS TO DELETE                     =
=================================================================*/
/*
If we have a global_storage object mapping:
{band_seed -> {band_signature -> [(file_id, line_num), ...]}}

this section, in parallel, builds a dict mapping: 
{file_id -> [line_num, line_num, line_num]}

And the iteration scheme is to iterate over each band seed in series,
but iterate over the keys of each {band signature -> [...]} in parallel

Note: the strategy here is NOT to build connected components, because there's questions about transitivity
*/

fn build_lines_to_kill(band_storage: &BandStorage) -> DashMap<usize, HashSet<usize>> 
// hashset because I think it's threadsafe if insert only (needs checking?)
{

    let pbar = build_pbar(band_storage.len(), "Bands");
    let lines_to_kill: DashMap<usize, HashSet<usize>>  = DashMap::new();

    for global_ref in band_storage.iter() {
        global_ref.value().par_iter().for_each(|entry| {
            let value = entry.value();
            for i in 1..value.len() {
                let (path_id, line_num) = &value[i];
                lines_to_kill
                    .entry(path_id.as_usize())
                    .or_default()
                    .insert(line_num.as_usize());
            }
        });
        pbar.inc(1);
    }
    lines_to_kill
}


fn save_lines_to_kill(lines_to_kill: DashMap<usize, HashSet<usize>>, path: &PathBuf) -> Result<(), Error>{
    // Saves the lines_to_kill object to a path
    let regular_map : HashMap<_,_> = lines_to_kill.into_par_iter().collect();
    let bytes = serde_json::to_vec(&regular_map).unwrap();
    write_mem_to_pathbuf(&bytes, &path).unwrap();
    Ok(())
}

fn load_lines_to_kill(path: &PathBuf) -> Result<HashMap<usize, HashSet<usize>>, Error>{
    let bytes = read_pathbuf_to_mem(path).unwrap().into_inner().into_inner();
    let data: HashMap<usize, HashSet<usize>> = serde_json::from_slice(&bytes).unwrap();
    Ok(data)
}

fn merge_lines_to_kill(all_lines_to_kill: Vec<HashMap<usize, HashSet<usize>>>) -> Result<DashMap<usize, HashSet<usize>>, Error> {
    let lines_to_kill: DashMap<usize, HashSet<usize>>  = DashMap::new();
    let _ = all_lines_to_kill
        .par_iter()
        .map(|dash| {
            dash
            .par_iter()
            .for_each(|(path_id, line_set)| {
                for line_num in line_set {
                    lines_to_kill.entry(*path_id).or_default()
                                 .insert(*line_num);
                }
            });
        }
    );
    Ok(lines_to_kill)
}


/*=================================================================
=                         WRITE OUTPUTS                           =
=================================================================*/
/*
Iterates over all paths seen, and removes the lines that we should from each
If there are no lines to kill, just copy to output
*/
fn scrub_all_paths(path_lookup: &PathLookup, lines_to_kill: DashMap<usize, HashSet<usize>>,
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


fn chop_lines(input_filename: &PathBuf, output_filename: &PathBuf, chop_lines: Option<HashSet<usize>>) -> Result<(usize, usize), Error> {
    let data = read_pathbuf_to_mem(input_filename).unwrap();
    let chop_lines: HashSet<usize> = if chop_lines == None {
        HashSet::new() 
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
=                                 MAIN                            =
=================================================================*/

fn main() {
    let args = Args::parse();
    minhash(args.input, args.output, args.num_bands, args.band_size, args.ngram_size, args.num_docs, args.max_lines_per_path).unwrap();
}

fn minhash(input: Vec<PathBuf>, output: PathBuf, num_bands: usize, band_size: usize, ngram_size: usize, num_docs: usize, max_lines_per_path: usize) -> Result<(), Error>{
    println!("Starting MinHash run...");    
    let start_main = Instant::now();
    // Phase 0: Setup, collect filenames, build path lookup, build band seeds
    let mut input_files = expand_dirs(input.clone()).unwrap();
    input_files.sort(); // sort before building the path lookup
    println!("Collected {:?} input files", input_files.len());
    let path_lookup = PathLookup::new(input_files.clone());
    let band_seeds: Vec<u64> = (0..num_bands).map(|i| i as u64).collect();


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




