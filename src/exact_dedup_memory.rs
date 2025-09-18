use anyhow::{anyhow, Error, Result};
use dashmap::DashMap;
use rayon::prelude::*;
use serde_json::{json, Value};
use std::fmt::Debug;
use std::hash::Hash;
use std::io::BufRead;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use xxhash_rust::xxh3::{xxh3_128, xxh3_64};

use crate::utils::{json_get, json_set};
use mj_io::{
    build_pbar, expand_dirs, get_output_filename, read_pathbuf_to_mem, write_mem_to_pathbuf,
};
use std::time::Instant;
/*
EXACT DEDUPLICATION MODULE

Fast, parallel deduplication of JSONL files using xxHash3 for duplicate detection.

## Usage
- Processes directories of JSON Lines files
- Removes duplicates based on text content or pre-computed hashes
- Supports 64-bit and 128-bit hash modes
- Uses parallel processing with thread-safe hash counting

## Parameters
- `text_key`: JSON field containing text content to deduplicate
- `hash_key`: Optional field with pre-computed hashes (skips hashing step)
- `hash_bits`: 64 or 128 bit hash size (64-bit sufficient for ~4B documents)
*/

/*=======================================================
=                     HELPER DATA TYPES                 =
=======================================================*/

trait DocHash: Copy + Eq + Hash + Debug + Send + Sync + 'static {
    fn hash_string(text: &String) -> Self;
    fn from_json(value: &Value) -> Result<Self, Error>;
    fn to_json(&self) -> Value;
}
impl DocHash for u64 {
    fn hash_string(text: &String) -> Self {
        xxh3_64(text.as_bytes())
    }
    fn from_json(value: &Value) -> Result<Self, Error> {
        Ok(value.as_u64().unwrap())
    }
    fn to_json(&self) -> Value {
        Value::Number((*self).into())
    }
}

impl DocHash for u128 {
    fn hash_string(text: &String) -> Self {
        xxh3_128(text.as_bytes())
    }
    fn from_json(value: &Value) -> Result<Self, Error> {
        println!("VALUE IS {:?}", value.as_str());
        Ok(value.as_str().unwrap().parse::<u128>().unwrap())
    }
    fn to_json(&self) -> Value {
        Value::String(self.to_string()) // u128 as string since JSON doesn't support 128-bit ints
    }
}

/*=====================================================
=                      MAIN FXN                       =
=====================================================*/

pub fn exact_dedup_memory(
    input_dir: &PathBuf,
    output_dir: &PathBuf,
    text_key: &String,
    hash_key: Option<String>,
    hash_bits: usize,
    annotate: Option<String>,
) -> Result<(), Error> {
    let start_main = Instant::now();
    println!("Starting exact deduplication");

    let (seen_docs, kept_docs) = match hash_bits {
        64 => exact_dedup_impl::<u64>(input_dir, output_dir, text_key, hash_key, annotate).unwrap(),
        128 => {
            exact_dedup_impl::<u128>(input_dir, output_dir, text_key, hash_key, annotate).unwrap()
        }
        _ => {
            return Err(anyhow!(
                "Hash bits can only be 64 or 128, not {:?}",
                hash_bits
            ));
        }
    };
    let removal_rate = 100.0 * ((seen_docs - kept_docs) as f32) / seen_docs as f32;
    println!(
        "Finished exact deduplication in {:?} seconds",
        start_main.elapsed().as_secs()
    );
    println!(
        "Saw {:?} documents, and kept {:?} of them",
        seen_docs, kept_docs
    );
    println!("Removal rate was {:.2}", removal_rate);

    Ok(())
}

fn exact_dedup_impl<K: DocHash>(
    input_dir: &PathBuf,
    output_dir: &PathBuf,
    text_key: &String,
    hash_key: Option<String>,
    annotate: Option<String>,
) -> Result<(usize, usize), Error> {
    let input_paths = expand_dirs(vec![input_dir.clone()], None).unwrap();
    println!("INPUT DIR {:?}", input_dir.clone());
    println!("INPUT FILES {:?}", input_paths);
    let seen_docs = AtomicUsize::new(0);
    let kept_docs = AtomicUsize::new(0);
    let counter: DashMap<K, usize> = DashMap::new();
    if let Some(ref _annotate_key) = annotate {
        let anno_pbar = build_pbar(input_paths.len(), "Paths");
        input_paths.par_iter().for_each(|p| {
            build_out_counter(p, text_key, &hash_key, &counter).unwrap();
            anno_pbar.inc(1);
        })
    };

    let pbar = build_pbar(input_paths.len(), "Paths");
    input_paths.into_par_iter().for_each(|p| {
        let output_filename = get_output_filename(&p, input_dir, output_dir).unwrap();
        let (p_seen, p_kept) =
            exact_dedup_file(p, output_filename, text_key, &hash_key, &counter, &annotate).unwrap();
        seen_docs.fetch_add(p_seen, Ordering::Relaxed);
        kept_docs.fetch_add(p_kept, Ordering::Relaxed);
        pbar.inc(1);
    });
    Ok((seen_docs.into_inner(), kept_docs.into_inner()))
}

fn build_out_counter<K: DocHash>(
    p: &PathBuf,
    text_key: &String,
    hash_key: &Option<String>,
    counter: &DashMap<K, usize>,
) -> Result<(), Error> {
    let data = read_pathbuf_to_mem(p).unwrap();
    for line in data.lines() {
        let line = line.unwrap();
        let line_json: Value = serde_json::from_str(&line).unwrap();
        let hash_val = get_hash_val::<K>(&line_json, text_key, hash_key).unwrap();
        counter.entry(hash_val).and_modify(|c| *c += 1).or_insert(1);
    }
    Ok(())
}

fn exact_dedup_file<K: DocHash>(
    p: PathBuf,
    output_filename: PathBuf,
    text_key: &String,
    hash_key: &Option<String>,
    counter: &DashMap<K, usize>,
    annotate: &Option<String>,
) -> Result<(usize, usize), Error> {
    let mut seen = 0;
    let mut kept = if let Some(_anno) = annotate {
        counter.len()
    } else { 
        0
    };

    let mut output_contents: Vec<u8> = Vec::new();

    let data = read_pathbuf_to_mem(&p).unwrap();
    for line in data.lines() {
        let line = line.unwrap();
        seen += 1;
        let mut line_json: Value = serde_json::from_str(&line).unwrap();
        let hash_val = get_hash_val::<K>(&line_json, text_key, hash_key).unwrap();
        if let Some(annotate_key) = annotate {
            let count = counter.get(&hash_val).unwrap();
            let anno_data = json!({"hash": hash_val.to_json(),
			 					   "num_dups": *count});
            json_set(&mut line_json, &annotate_key, anno_data).unwrap();
            output_contents.extend(serde_json::to_vec(&line_json).unwrap());
            output_contents.push(b'\n');
        } else {
            let count = *counter.entry(hash_val).and_modify(|c| *c += 1).or_insert(1);
            if count == 1 {
                kept += 1;
                output_contents.extend(line.into_bytes());
                output_contents.push(b'\n');
            }
        }
    }
    if output_contents.len() > 0 {
        write_mem_to_pathbuf(&output_contents, &output_filename).unwrap()
    }



    Ok((seen, kept))
}

fn get_hash_val<K: DocHash>(
    json_obj: &Value,
    text_key: &String,
    hash_key: &Option<String>,
) -> Result<K, Error> {
    let hash_val: K = if let Some(key) = hash_key {
        // type of this should be K
        let hash_val_value = json_get(json_obj, key).unwrap();
        DocHash::from_json(hash_val_value).unwrap()
    } else {
        let text = json_obj
            .get(text_key)
            .unwrap()
            .as_str()
            .unwrap()
            .to_string();
        DocHash::hash_string(&text)
    };
    Ok(hash_val)
}
