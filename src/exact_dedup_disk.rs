//! # Disk-Based Exact Deduplication
//!
//! Multi-node parallelizable exact deduplication for datasets too large to fit in memory.
//!
//! ## Overview
//!
//! This module implements a two-phase approach to exact deduplication that enables
//! distributed processing across multiple machines:
//!
//! ### Phase 1: Group (Partitioning)
//! Documents are hashed and organized into bins based on their hash values. Each
//! bin is written to disk as a separate file. This allows the dataset to be
//! partitioned across machines.
//!
//! ### Phase 2: Prune (Deduplication)
//! Each bin is loaded into memory independently and deduplicated. Since documents
//! with the same hash always end up in the same bin, we guarantee that all duplicates
//! are found.
//!
//! ## Distributed Processing Strategy
//!
//! ```text
//! Initial State (Documents organized by file):
//! ┌─────────────────┬───────┬───────┬───────┬───────┬───────┐
//! │ Files \ Hashes  │ doc1  │ doc2  │ doc3  │ doc4  │ doc5  │
//! ├─────────────────┼───────┼───────┼───────┼───────┼───────┤
//! │ path1.jsonl.zst │   X   │       │   X   │       │       │
//! │ path2.jsonl.zst │       │   X   │       │   X   │       │
//! │ path3.jsonl.zst │   X   │       │       │       │   X   │
//! └─────────────────┴───────┴───────┴───────┴───────┴───────┘
//!
//! After Phase 1 (Documents organized by hash bin):
//! ┌──────────────┬─────────────────────────┐
//! │ Bin 0        │ All docs with hash % n = 0 │
//! │ Bin 1        │ All docs with hash % n = 1 │
//! │ Bin 2        │ All docs with hash % n = 2 │
//! │ ...          │ ...                        │
//! └──────────────┴─────────────────────────┘
//! ```
//!
//! ## Workflow
//!
//! 1. **Phase 1**: Run `exact_dedup_disk_group` on each machine with local data
//! 2. **Shuffle**: Redistribute bins so all chunks of bin N are on the same machine
//! 3. **Phase 2**: Run `exact_dedup_disk_prune` on each machine to deduplicate its bins
//!
//! ## Choosing Number of Bins
//!
//! - More bins = better load balancing but more files
//! - Recommended: 100-1000 bins depending on dataset size
//! - Each bin should fit comfortably in memory during Phase 2

use crate::storage::GenWriter;
use crate::utils::{json_get, json_set};
use anyhow::{anyhow, Error, Result};
use dashmap::DashMap;
use mj_io::{
    build_pbar, expand_dirs, get_output_filename, read_pathbuf, create_writer
};
use rand::Rng;
use rayon::prelude::*;
use regex::Regex;
use serde_json::{json, Value};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use xxhash_rust::xxh3::{xxh3_128, xxh3_64};

/*======================================================================
=                            GROUP METHODS                             =
======================================================================*/

const MAX_SIZE: usize = 256_000_000;

/// **Phase 1**: Groups documents into bins based on hash values.
///
/// Reads all documents in `input_dir`, computes or reads their hash values,
/// and writes them to bin files in `storage_dir`. Documents with the same
/// hash always go to the same bin, enabling distributed deduplication.
///
/// # Arguments
///
/// * `input_dir` - Directory containing input JSONL files
/// * `storage_dir` - Working directory where bin files will be written
/// * `text_key` - JSON key containing document text (used if hash not present)
/// * `hash_key` - JSON key for hash values (will be added if not present)
/// * `hash_bits` - Hash size: 64 or 128 bits
/// * `num_bins` - Number of bins to partition documents into
///
/// # Output Structure
///
/// Creates files in `storage_dir`:
/// ```text
/// storage_dir/
/// └── chunk_00000000.{run_id}.jsonl.zst
/// └── chunk_00000001.{run_id}.jsonl.zst
/// └── ...
/// └── chunk_{num_bins-1}.{run_id}.jsonl.zst
/// ```
///
/// Where `run_id` is a unique identifier based on input files and random salt.
/// Files may be split if they exceed MAX_SIZE (256MB).
///
/// # Hash Key Behavior
///
/// - If document already has `hash_key` field: Uses existing value
/// - If `hash_key` missing: Computes hash from `text_key` and adds it to document
///
/// Format:
/// - 64-bit: JSON number
/// - 128-bit: JSON string (decimal)
///
/// # Parallelism
///
/// Multiple workers can run this function concurrently on different machines,
/// each processing their local portion of the dataset. All workers should use
/// the same `num_bins` value.
///
/// # Example
///
/// ```no_run
/// // Worker 1 processes /data/shard1
/// exact_dedup_disk_group(
///     &PathBuf::from("/data/shard1"),
///     &PathBuf::from("/scratch/bins"),
///     &"text".to_string(),
///     &"doc_hash".to_string(),
///     64,
///     100  // 100 bins
/// )?;
///
/// // Worker 2 processes /data/shard2 (same num_bins!)
/// exact_dedup_disk_group(
///     &PathBuf::from("/data/shard2"),
///     &PathBuf::from("/scratch/bins"),
///     &"text".to_string(),
///     &"doc_hash".to_string(),
///     64,
///     100  // Must match!
/// )?;
/// ```
///
/// # Returns
///
/// `Ok(())` on success, printing the number of documents processed.
///
/// # Notes
///
/// - All documents with the same hash go to the same bin number
/// - Bin assignment: `bin = hash % num_bins`
/// - After this phase, shuffle bins so all parts of each bin live on one machine
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

/// Processes a single file, assigning each document to a bin.
///
/// For each document:
/// 1. Read or compute hash value
/// 2. Add hash to document if not present
/// 3. Determine bin number: `hash % num_bins`
/// 4. Write to appropriate bin file
pub fn group_docs(
    p: &PathBuf,
    text_key: &String,
    hash_key: &String,
    hash_bits: usize,
    gen_writer: &GenWriter,
    num_bins: usize,
) -> Result<usize, Error> {
    let contents = read_pathbuf(p, true).unwrap();
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
        gen_writer.write_line(0, line, Some(bin_number)).unwrap();

        num_docs += 1;
    }

    Ok(num_docs)
}

/*======================================================================
=                            PRUNE METHODS                             =
======================================================================*/

/// **Phase 2**: Deduplicates documents within each bin.
///
/// Reads all bin files from `storage_dir` (created by Phase 1), groups them
/// by bin number, and deduplicates each group. Since all documents with the
/// same hash are in the same bin, this guarantees complete deduplication.
///
/// # Arguments
///
/// * `storage_dir` - Directory containing bin files from Phase 1
/// * `output_dir` - Directory where deduplicated files will be written
/// * `hash_key` - JSON key containing hash values (must match Phase 1)
/// * `annotate_key` - Optional key for adding duplicate metadata instead of removing
///
/// # Input Files
///
/// Expects files matching pattern: `chunk_{bin_number}.*.jsonl.zst`
///
/// All files with the same bin number are processed together as one group.
/// This allows Phase 1 to split large bins across multiple files.
///
/// # Annotation Format
///
/// When `annotate_key` is Some:
/// ```json
/// {
///   "your_annotation_key": {
///     "hash": "12345678901234567890",
///     "num_dups": 3
///   }
/// }
/// ```
///
/// # Parallelism
///
/// Groups (bins) are processed in parallel, but all files within a group
/// must be accessible from the same machine.
///
/// # Example
///
/// ```no_run
/// // After shuffling bins from multiple workers in Phase 1
/// exact_dedup_disk_prune(
///     &PathBuf::from("/scratch/bins"),
///     &PathBuf::from("/data/deduplicated"),
///     &"doc_hash".to_string(),
///     &None  // Remove duplicates
/// )?;
///
/// // Or with annotation
/// exact_dedup_disk_prune(
///     &PathBuf::from("/scratch/bins"),
///     &PathBuf::from("/data/annotated"),
///     &"doc_hash".to_string(),
///     &Some("duplicate_info".to_string())
/// )?;
/// ```
///
/// # Returns
///
/// `Ok(())` on success, printing statistics:
/// - Total documents processed
/// - Documents kept (unique documents)
/// - Removal rate percentage
///
/// # Memory Requirements
///
/// Each bin must fit in memory during processing. If Phase 1 created bins
/// that are too large, increase `num_bins` and rerun.
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

/// Deduplicates a single bin (group of files with same bin number).
///
/// Loads all files in the group into memory, deduplicates based on hash values,
/// and writes output. Since all documents with the same hash are guaranteed to
/// be in this bin, we find all duplicates.
///
/// # Algorithm
///
/// 1. If annotating: Pre-count occurrences of each hash
/// 2. Process files, keeping first occurrence of each hash
/// 3. If annotating: Add metadata to all documents
/// 4. Write output files
///
/// # Returns
///
/// Tuple of (documents_seen, documents_kept)
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
            let contents = read_pathbuf(p, true).unwrap();
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
        let contents = read_pathbuf(p, true).unwrap();        
        let output_filename = get_output_filename(&p, storage_dir, output_dir).unwrap();
        let mut writer = create_writer(&output_filename).unwrap();
        for line in contents.lines() {
            let line = line.unwrap();
            let mut line_json = serde_json::from_str(&line).unwrap();
            let hash_val = json_get(&line_json, hash_key).unwrap();
            if let Some(anno) = annotate_key {
                let count = counter.get(hash_val).unwrap();
                let anno_data = json!({"hash": hash_val,
			 					       "num_dups": *count});
                json_set(&mut line_json, &anno, anno_data).unwrap();
                writer.write_line(&serde_json::to_vec(&line_json).unwrap()).unwrap();
            } else {
                let count = *counter
                    .entry(hash_val.clone())
                    .and_modify(|c| *c += 1)
                    .or_insert(1);
                if count == 1 {
                    writer.write_line(&serde_json::to_vec(&line_json).unwrap()).unwrap();
                }
            }
        }
        writer.finish().unwrap();
    });
    let kept_docs = counter.len();
    let docs_seen = counter.into_par_iter().map(|(_k, v)| v).sum::<usize>();

    Ok((docs_seen, kept_docs))
}
