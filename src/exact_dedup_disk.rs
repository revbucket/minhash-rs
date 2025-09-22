use crate::storage::GenWriter;
use crate::utils::{json_get, json_set};
use anyhow::{anyhow, Error, Result};
use dashmap::DashMap;
use mj_io::{
    build_pbar, expand_dirs, get_output_filename, read_pathbuf_to_mem, write_mem_to_pathbuf,
};
use rand::Rng;
use rayon::prelude::*;
use regex::Regex;
use serde_json::{json, Value};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::io::BufRead;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use xxhash_rust::xxh3::{xxh3_128, xxh3_64};
/*
Disk-based multi-node parallelizable exact deduplication in rust. This should only be used
in the case that the size of the dataset is too large to fit on a single node.

This operates in two phases:

Phase 1 (group):
- Reads and annotates documents with their hash signatures (if not already present)
- Then reorganizes documents into "bins" on disk based on their hash signatures

Phase 2:
- Loops over each bin and loads it fully into memory.
- Keeps exactly one copy of each document with a given hash signature in that bin.



The current organization of docs can be thought of like:
+---------------------+-------+-------+-------+-------+-------+
| Paths \ Hash Sigs   | doc1  | doc2  | doc3  | doc4  | doc5  |
+---------------------+-------+-------+-------+-------+-------+
| path1.jsonl.zst     |       |       |       |       |       |
+---------------------+-------+-------+-------+-------+-------+
| path2.jsonl.zst     |       |       |       |       |       |
+---------------------+-------+-------+-------+-------+-------+
| path3.jsonl.zst     |       |       |       |       |       |
+---------------------+-------+-------+-------+-------+-------+
| path4.jsonl.zst     |       |       |       |       |       |
+---------------------+-------+-------+-------+-------+-------+
| path5.jsonl.zst     |       |       |       |       |       |
+---------------------+-------+-------+-------+-------+-------+

Where paths may be split across various nodes.
The idea is that on each node containing data, we break docs into bins based on their hash signature (step 1)
Then we shuffle bins around so that all bins corresponding to hashes starting with 0x1234 (e.g.) live on the same node
Then we apply step 2 to all nodes

## Usage

*/

/*======================================================================
=                            GROUP METHODS                             =
======================================================================*/

const MAX_SIZE: usize = 256_000_000;

pub fn exact_dedup_disk_group(
    input_dir: &PathBuf,
    storage_dir: &PathBuf,
    text_key: &String,
    hash_key: &String,
    hash_bits: usize,
    num_bins: usize,
) -> Result<(), Error> {
    let start_main = Instant::now();
    let docs_seen = AtomicUsize::new(0);
    println!("Starting grouping operation...");

    let mut input_paths = expand_dirs(vec![input_dir.to_path_buf()], None).unwrap();
    input_paths.sort();
    let mut hasher = DefaultHasher::new();
    for path in &input_paths {
        path.hash(&mut hasher);
    }
    let hash_fingerprint = hasher.finish();
    let mut rng = rand::thread_rng();
    let salt: String = (0..1).map(|_| format!("{:02x}", rng.gen::<u8>())).collect();
    let run_id = format!("{:08x}_{:}", hash_fingerprint as u32, salt);
    let gen_writer = GenWriter::new(
        storage_dir,
        num_bins,
        &run_id,
        &Some(String::from("jsonl.zst")),
        true,
        Some(MAX_SIZE),
    );

    let pbar = build_pbar(input_paths.len(), "Paths");
    input_paths.into_par_iter().for_each(|p| {
        docs_seen.fetch_add(
            group_docs(&p, text_key, hash_key, hash_bits, &gen_writer, num_bins).unwrap(),
            Ordering::Relaxed,
        );
        pbar.inc(1);
    });
    gen_writer.finish().unwrap();

    let docs_seen = docs_seen.into_inner();
    println!(
        "Grouped {:?} docs in {:?} seconds",
        docs_seen,
        start_main.elapsed().as_secs()
    );
    Ok(())
}

pub fn group_docs(
    p: &PathBuf,
    text_key: &String,
    hash_key: &String,
    hash_bits: usize,
    gen_writer: &GenWriter,
    num_bins: usize,
) -> Result<usize, Error> {
    let contents = read_pathbuf_to_mem(p).unwrap();
    let mut num_docs = 0;
    for line in contents.lines() {
        let mut line = line.unwrap().into_bytes();
        let mut json_line: Value = serde_json::from_slice(&line).unwrap();
        let hash_val_opt = json_get(&json_line, hash_key);
        let bin_number = if let Some(hash_value) = hash_val_opt {
            let bin_number = match hash_bits {
                64 => (hash_value.as_u64().unwrap() as usize) % num_bins,
                128 => {
                    let hash_value128: u128 = hash_value.as_str().unwrap().parse::<u128>().unwrap();
                    (hash_value128 % (num_bins as u128)) as usize
                }
                _ => {
                    return Err(anyhow!(
                        "Hash bits can only be 64 or 128, not {:?}",
                        hash_bits
                    ))
                }
            };
            bin_number
        } else {
            let text = json_get(&json_line, text_key)
                .unwrap()
                .as_str()
                .unwrap()
                .to_string();
            let bin_number = match hash_bits {
                64 => {
                    let hash_value = xxh3_64(text.as_bytes());
                    json_set(&mut json_line, &hash_key, json!(hash_value)).unwrap();
                    hash_value as usize % num_bins
                }
                128 => {
                    let hash_value = xxh3_128(text.as_bytes());
                    json_set(&mut json_line, &hash_key, json!(hash_value.to_string())).unwrap();
                    (hash_value % (num_bins as u128)) as usize
                }
                _ => {
                    return Err(anyhow!(
                        "Hash bits can only be 64 or 128, not {:?}",
                        hash_bits
                    ))
                }
            };
            bin_number
        };
        line = serde_json::to_vec(&json_line).unwrap();        
        line.push(b'\n');
        gen_writer.write_line(0, line, bin_number).unwrap();

        num_docs += 1;
    }

    Ok(num_docs)
}

/*======================================================================
=                            PRUNE METHODS                             =
======================================================================*/

pub fn exact_dedup_disk_prune(
    storage_dir: &PathBuf,
    output_dir: &PathBuf,
    hash_key: &String,
    annotate_key: &Option<String>,
) -> Result<(), Error> {
    let start_main = Instant::now();
    println!("Starting pruning operation...");
    let docs_seen = AtomicUsize::new(0);
    let docs_kept = AtomicUsize::new(0);
    let input_paths = expand_dirs(vec![storage_dir.to_path_buf()], None).unwrap();
    // All should have naming conventions like chunk_{:08}.{}.bin
    let re = Regex::new(r"chunk_(\d{8})\.").unwrap();
    let mut groups: HashMap<usize, Vec<PathBuf>> = HashMap::new();
    for p in input_paths {
        let base_name = p.file_name().unwrap().to_str().unwrap();
        let bin_num = re
            .captures(base_name)
            .unwrap()
            .get(1)
            .unwrap()
            .as_str()
            .parse::<usize>()
            .unwrap();
        groups.entry(bin_num).or_default().push(p);
    }
    let pbar = build_pbar(groups.len(), "Groups");
    groups.values().into_iter().for_each(|vlist| {
        let (p_docs_seen, p_docs_kept) =
            prune_group(vlist, storage_dir, output_dir, hash_key, annotate_key).unwrap();
        docs_seen.fetch_add(p_docs_seen, Ordering::Relaxed);
        docs_kept.fetch_add(p_docs_kept, Ordering::Relaxed);
        pbar.inc(1);
    });
    let seen_docs = docs_seen.into_inner();
    let kept_docs = docs_kept.into_inner();

    let removal_rate = 100.0 * ((seen_docs - kept_docs) as f32) / seen_docs as f32;
    println!(
        "Finished exact deduplication in {:?} seconds",
        start_main.elapsed().as_secs()
    );
    println!(
        "Saw {:?} documents, and kept {:?} of them",
        seen_docs, kept_docs
    );
    println!("Removal rate was {:.2}%", removal_rate);
    Ok(())
}

fn prune_group(
    vlist: &Vec<PathBuf>,
    storage_dir: &PathBuf,
    output_dir: &PathBuf,
    hash_key: &String,
    annotate_key: &Option<String>,
) -> Result<(usize, usize), Error> {
    let counter: DashMap<Value, usize> = DashMap::new(); // Maps hash key -> usize
    if let Some(_anno) = annotate_key {
        vlist.par_iter().for_each(|p| {
            let contents = read_pathbuf_to_mem(p).unwrap();
            for line in contents.lines() {
                let line = line.unwrap();
                let line_json = serde_json::from_str(&line).unwrap();
                let hash_val = json_get(&line_json, hash_key).unwrap();
                counter
                    .entry(hash_val.clone())
                    .and_modify(|c| *c += 1)
                    .or_insert(1);
            }
        })
    }
    vlist.par_iter().for_each(|p| {
        let contents = read_pathbuf_to_mem(p).unwrap();
        let mut output_contents: Vec<u8> = Vec::new();
        for line in contents.lines() {
            let line = line.unwrap();
            let mut line_json = serde_json::from_str(&line).unwrap();
            let hash_val = json_get(&line_json, hash_key).unwrap();
            if let Some(anno) = annotate_key {
                let count = counter.get(hash_val).unwrap();
                let anno_data = json!({"hash": hash_val,
			 					       "num_dups": *count});
                json_set(&mut line_json, &anno, anno_data).unwrap();
                output_contents.extend(serde_json::to_vec(&line_json).unwrap());
                output_contents.push(b'\n');
            } else {
                let count = *counter
                    .entry(hash_val.clone())
                    .and_modify(|c| *c += 1)
                    .or_insert(1);
                if count == 1 {
                    output_contents.extend(line.into_bytes());
                    output_contents.push(b'\n');
                }
            }
        }
        if output_contents.len() > 0 {
            let output_filename = get_output_filename(&p, storage_dir, output_dir).unwrap();
            write_mem_to_pathbuf(&output_contents, &output_filename).unwrap();
        }
    });
    let kept_docs = counter.len();
    let docs_seen = counter.into_par_iter().map(|(_k, v)| v).sum::<usize>();

    Ok((docs_seen, kept_docs))
}
