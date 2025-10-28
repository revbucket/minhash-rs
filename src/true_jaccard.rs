use std::fs;
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{BuildHasher, Hash, Hasher};
use std::io::BufRead;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;

use ahash::RandomState;
use anyhow::{Error, Result};
use dashmap::DashMap;
use mj_io::{build_pbar, expand_dirs, read_pathbuf_to_mem, write_mem_to_pathbuf};
use rayon::prelude::*;
use regex::Regex;
use serde_json::{json, Value as JSONValue};

use crate::minhash_base::{preprocess_text, OmniTokenizer};
use crate::minhash_config::{
    Config, ConfigOverrides, EngOverrides, MinHashOverrides, OutputOverrides,
};
use crate::uf_rush2::{parent as uf_parent, UFRush};
use crate::utils::{json_get, json_set};

const OUTPUT_FILE_SIZE: usize = 256_000_000;

pub fn true_jaccard(
    input_dir: &PathBuf,
    output_dir: &PathBuf,
    minhash_cc_id: Option<String>,
    group_regex: Option<String>,
    config: Option<PathBuf>,
    jaccard_threshold: f64,
    ngram_size: Option<usize>,
    tokenizer: Option<String>,
    annotate_key: &str,
    hotnode_size: Option<usize>,
    hotnode_dir: Option<PathBuf>,
    parallel_nest: usize,
    id_offset: Option<usize>,
    delete_while_cleaning: Option<bool>,
) -> Result<()> {
    let start_main = Instant::now();
    println!("Starting true jaccard checks");

    // Load the config if present (to get document->ngramSet params)
    let overrides = ConfigOverrides {
        minhash_params: MinHashOverrides {
            num_buckets: None,
            bucket_size: None,
            ngram_size,
            permutation_seed: None,
            tokenizer,
        },
        eng_params: EngOverrides {
            num_docs: None,
            max_lines_per_path: None,
            num_sig_chunks: None,
        },
        output_params: OutputOverrides {
            annotate: Some(true),
            annotate_key: Some(annotate_key.to_string()),
            delete_while_cleaning: Some(false),
            remove_duplicates: Some(false),
        },
    };
    let delete_while_cleaning = delete_while_cleaning.unwrap_or(overrides.output_params.delete_while_cleaning.unwrap());

    let hotnode_size = hotnode_size.unwrap_or(usize::MAX);
    let hotnode_dir = hotnode_dir.unwrap_or_else(|| output_dir.clone());
    let id_offset = id_offset.unwrap_or_else(|| 0);
    let config_obj = Config::load_with_overrides(config, overrides).unwrap();
    let ngram_size = config_obj.minhash_params.ngram_size;
    let tokenizer_name = config_obj.minhash_params.tokenizer;
    let tokenizer = OmniTokenizer::new(&tokenizer_name).unwrap();
    // Gather files into groups
    let paths = expand_dirs(vec![input_dir.clone()], None).unwrap();
    let input_groups: Vec<Vec<PathBuf>> = if let Some(group_regex) = group_regex {
        let mut group_hash: HashMap<Option<String>, Vec<PathBuf>> = HashMap::new();
        let re = Regex::new(&group_regex).unwrap();
        for p in &paths {
            let g = p
                .file_name()
                .and_then(|name| name.to_str())
                .and_then(|name| re.find(name))
                .map(|m| m.as_str().to_string());
            group_hash.entry(g).or_default().push(p.clone());
        }
        group_hash.into_values().collect()
    } else {
        vec![paths.clone()]
    };

    let hotnode_counter = AtomicUsize::new(0);
    let output_counter = AtomicUsize::new(0);
    let new_cc_counter = AtomicUsize::new(id_offset);
    let total_count = AtomicUsize::new(0);
    let hotnode_count = AtomicUsize::new(0);
    let remove_count = AtomicUsize::new(0);

    // Loop over each "group" and calculate exact jaccard similarities
    let pbar = build_pbar(paths.len(), "Paths");
    let chunk_size = (paths.len() - 1) / parallel_nest + 1;

    input_groups.par_chunks(chunk_size).for_each(|chunk| {
        for pvec in chunk {
            let (group_total, group_hotnode, group_remove) = true_jacc_group(
                &pvec,
                output_dir,
                minhash_cc_id.clone(),
                jaccard_threshold,
                ngram_size,
                &tokenizer,
                hotnode_size,
                &hotnode_counter,
                &hotnode_dir,
                annotate_key.to_string(),
                &output_counter,
                &new_cc_counter,
                &delete_while_cleaning,
                input_dir,
            )
            .unwrap();
            total_count.fetch_add(group_total, Ordering::SeqCst);
            hotnode_count.fetch_add(group_hotnode, Ordering::SeqCst);
            remove_count.fetch_add(group_remove, Ordering::SeqCst);
            pbar.inc(pvec.len() as u64);
        }
    });

    let total_count = total_count.into_inner();
    let hotnode_count = hotnode_count.into_inner();
    let remove_count = remove_count.into_inner();

    println!(
        "Finished true jaccard checks in {} secs",
        start_main.elapsed().as_secs()
    );
    println!(
        "Saw {} docs | Removed {} hotnode docs | Removed {:?} docs | Removal rate would {:.2}%",
        total_count,
        hotnode_count,
        remove_count,
        100.0 * (remove_count as f64) / ((total_count - hotnode_count) as f64)
    );
    Ok(())
}

fn true_jacc_group(
    pvec: &Vec<PathBuf>,
    output_dir: &PathBuf,
    minhash_cc_id: Option<String>,
    jaccard_threshold: f64,
    ngram_size: usize,
    tokenizer: &OmniTokenizer,
    hotnode_size: usize,
    hotnode_counter: &AtomicUsize,
    hotnode_dir: &PathBuf,
    annotate_key: String,
    output_counter: &AtomicUsize,
    new_cc_counter: &AtomicUsize,
    delete_while_cleaning: &bool,
    input_dir: &PathBuf,
) -> Result<(usize, usize, usize), Error> {
    // Handles a group of files to filter for jaccard similarity between all pairs that share a 'minhash' (or all, if none have)

    // Step 0: Load all docs in group into memory (flatmap load them)
    // Parallel across documents

    let all_docs: Vec<JSONValue> = pvec
        .par_iter()
        .flat_map(|p| {
            let contents = read_pathbuf_to_mem(&p).unwrap();
            contents
                .lines()
                .map(|line| {
                    let line = line.unwrap();
                    let doc: JSONValue = serde_json::from_str(&line).unwrap();
                    doc
                })
                .collect::<Vec<JSONValue>>()
        })
        .collect();
    if *delete_while_cleaning {
        pvec.par_iter().for_each(|p| {
            fs::remove_file(&input_dir.clone().join(p)).unwrap();
        });
    }
    let n = all_docs.len();
    let output_docs = Arc::new(Mutex::new(Vec::new()));

    // Step 1: Group by previously existing cc_id (count all 'missing' cc_id's together)
    // Parallel across documents
    let groups: DashMap<String, Vec<JSONValue>> = DashMap::new();
    if let Some(minhash_cc_id) = minhash_cc_id {
        all_docs.into_par_iter().for_each(|v| {
            let cc_id_response = json_get(&v, &minhash_cc_id);
            // If minhash_cc_id is specified, and present, group according to it
            if let Some(cc_id) = cc_id_response {
                groups.entry(cc_id.to_string()).or_default().push(v);
            } else {
                // If specified, just ignore these docs (but make sure they get written)
                let mut output_guard = output_docs.lock().unwrap();
                output_guard.push(v);
            }
        });
    } else {
        groups
            .entry(String::from("None"))
            .or_default()
            .extend(all_docs)
    };
    let mut output_docs = output_docs.lock().unwrap();

    // Step 2: Split off all "hot nodes" and write them
    // (Parallel across groups)
    let mut group_freq: HashMap<usize, usize> = HashMap::new();
    groups.iter().for_each(|entry| {
        let len = entry.value().len();
        let cur_val = group_freq.get(&len).unwrap_or(&0);
        group_freq.insert(len, cur_val + 1);
        //let cur_val =
        //group_freq.
        //group_freq.entry(len).and_modify(|c| *c += 1);
    });

    let (hotnodes, proc_groups): (Vec<_>, Vec<_>) = groups
        .into_par_iter()
        .map(|(_k, v)| v)
        .partition(|v| v.len() >= hotnode_size);
    let hotnode_count = hotnodes.iter().map(|v| v.len()).sum();
    let hotnode_docs: Vec<JSONValue> = hotnodes.into_par_iter().flat_map_iter(|v| v).collect();
    write_docs(hotnode_docs, hotnode_counter, hotnode_dir, "hotnode").unwrap();

    // Step 3: Make token-ngram sets, gather indices to check, and check jaccard similarities
    // (parallel across docs, then pairs of docs)
    let toksets = toksetify(&proc_groups, tokenizer, ngram_size).unwrap();
    let pair_indices = generate_pair_indices::<HashSet<u64>>(&toksets);
    let pbar = build_pbar(pair_indices.len(), "Pair checks");
    let passing_pairs: Vec<&(usize, usize, usize)> = pair_indices
        .par_iter()
        .filter(|(g, i, j)| {
            let hashset_i = &toksets[*g][*i];
            let hashset_j = &toksets[*g][*j];
            let (hashset_i, hashset_j) = if hashset_i.len() < hashset_j.len() {
                (hashset_i, hashset_j)
            } else {
                (hashset_j, hashset_i)
            };
            let intersection_size: usize = hashset_i
                .iter()
                .map(|v| if hashset_j.contains(&v) { 1 } else { 0 })
                .sum();
            let union_size = hashset_i.len() - intersection_size + hashset_j.len(); // p.i.e
            let jacc_score = intersection_size as f64 / union_size as f64;
            pbar.inc(1);
            if jacc_score >= jaccard_threshold {
                true
            } else {
                false
            }
        })
        .collect();
    // Step 4: Take passing pairs/edges and enter into a UnionFind structure to get CC's
    // (Parallel everywhere)

    let uf = UFRush::new();

    passing_pairs.into_par_iter().for_each(|(g, i, j)| {
        let idx_i = n * g + i;
        let idx_j = n * g + j;
        uf.unite(idx_i, idx_j);
    });
    // And then compress all paths in the union find
    let keys: Vec<usize> = uf.nodes.par_iter().map(|entry| *entry.key()).collect();
    keys.into_par_iter().for_each(|k| {
        uf.find_path_compression(k);
    });

    // Compute cc_sizes
    let cc_sizes: DashMap<usize, usize> = DashMap::new(); // parent_id -> cc_size

    uf.nodes.par_iter().for_each(|entry| {
        let val = entry.value().load(Ordering::Relaxed);
        let uf_cc_id = uf_parent(val);
        cc_sizes
            .entry(uf_cc_id)
            .and_modify(|count| *count += 1)
            .or_insert(1);
    });
    // Compute global cc_ids
    let cc_id_lookup: DashMap<usize, usize> = DashMap::new(); // parent_id -> (global) cc_id
    cc_sizes.par_iter().for_each(|entry| {
        let new_cc_id = new_cc_counter.fetch_add(1, Ordering::Relaxed);
        cc_id_lookup.insert(*entry.key(), new_cc_id);
    });

    // Step 5: Annotate the docs and write to new directory
    let (annotated_docs, remove_count) =
        par_annotate(proc_groups, n, cc_sizes, cc_id_lookup, &uf, &annotate_key).unwrap();
    output_docs.extend(annotated_docs);

    write_docs(output_docs.to_vec(), output_counter, output_dir, "chunk").unwrap();
    Ok((n, hotnode_count, remove_count))
}

fn text_2_tokset(
    text: &String,
    tokenizer: &OmniTokenizer,
    ngram_size: usize,
) -> Result<HashSet<u64>, Error> {
    let mut tokset: HashSet<u64> = HashSet::new();
    let tokens = preprocess_text(&text.as_str(), tokenizer);
    let mut ngram: VecDeque<usize> = VecDeque::with_capacity(ngram_size);
    let mut ngram_count = 0;

    for tok in tokens {
        ngram.push_back(tok);
        if ngram.len() >= ngram_size {
            ngram_count += 1;
            let builder = RandomState::with_seeds(1, 2, 3, 4);
            let mut hasher = builder.build_hasher();
            ngram.hash(&mut hasher);
            let hash_val = hasher.finish();
            tokset.insert(hash_val);
            ngram.pop_front();
        }
    }
    if ngram_count == 0 {
        let builder = RandomState::with_seeds(1, 2, 3, 4);
        let mut hasher = builder.build_hasher();
        ngram.hash(&mut hasher);
        let hash_val = hasher.finish();
        tokset.insert(hash_val);
    }

    Ok(tokset)
}

fn toksetify(
    docs: &Vec<Vec<JSONValue>>,
    tokenizer: &OmniTokenizer,
    ngram_size: usize,
) -> Result<Vec<Vec<HashSet<u64>>>, Error> {
    // Flatten with group indices
    let flat_with_indices: Vec<(usize, &JSONValue)> = docs
        .par_iter()
        .enumerate()
        .flat_map_iter(|(group_idx, inner_vec)| inner_vec.iter().map(move |obj| (group_idx, obj)))
        .collect();

    // Process in parallel with good load balancing
    let processed: Vec<(usize, HashSet<u64>)> = flat_with_indices
        .par_iter()
        .map(|(group_idx, obj)| {
            let text = obj["text"].as_str().unwrap().to_string();
            (
                *group_idx,
                text_2_tokset(&text, tokenizer, ngram_size).unwrap(),
            )
        })
        .collect();

    // Reconstruct groups
    let mut result: Vec<Vec<HashSet<u64>>> = vec![Vec::new(); docs.len()];
    for (group_idx, processed_obj) in processed {
        result[group_idx].push(processed_obj);
    }

    Ok(result)
}

fn par_annotate(
    docs: Vec<Vec<JSONValue>>,
    n: usize,
    cc_size: DashMap<usize, usize>,
    cc_id_lookup: DashMap<usize, usize>,
    uf: &UFRush,
    annotate_key: &String,
) -> Result<(Vec<JSONValue>, usize), Error> {
    // Pairs are like [(uf_index/id, doc), ...]     
    let flat_with_indices: Vec<(usize, JSONValue)> = docs 
        .into_par_iter()
        .enumerate()
        .flat_map_iter(|(group_idx, inner_vec)| {
            inner_vec
                .into_iter()
                .enumerate()
                .map(move |(i, obj)| (n * group_idx + i, obj))
                .collect::<Vec<(usize, JSONValue)>>()
        })
        .collect();

    // Collect parents in a vector format (matches index w/ flat_with_indices, but has the doc parent)
    let parent_lookup: Vec<Option<usize>> = flat_with_indices
        .par_iter()
        .map(|(i, _)| {
            uf.nodes.get(i).map(|uf_val| {
                let val = uf_val.value().load(Ordering::Relaxed);
                uf_parent(val)
            })
        })
        .collect();

    // Build (parent, doc_idx) pairs in parallel
    let parent_doc_pairs: Vec<(usize, usize)> = parent_lookup
        .par_iter()
        .enumerate()
        .filter_map(|(doc_idx, parent_opt)| {
            parent_opt.map(|parent| (parent, doc_idx))
        })
        .collect();

    // Group by parent using parallel fold + reduce (no race conditions!)
    let parent_groups: HashMap<usize, Vec<usize>> = parent_doc_pairs
        .into_par_iter()
        .fold(
            || HashMap::new(),
            |mut acc: HashMap<usize, Vec<usize>>, (parent, doc_idx)| {
                acc.entry(parent).or_default().push(doc_idx);
                acc
            }
        )
        .reduce(
            || HashMap::new(),
            |mut acc1, acc2| {
                for (parent, mut docs) in acc2 {
                    acc1.entry(parent).or_default().append(&mut docs);
                }
                acc1
            }
        );

    // Map doc_idx -> cc_idx
    let cc_idx_array: DashMap<usize, usize> = DashMap::new();
    parent_groups.into_par_iter().for_each(|(_parent, mut indices)| {
        // Sort to ensure consistent ordering
        indices.sort_unstable();
        for (cc_idx, &doc_idx) in indices.iter().enumerate() {
            cc_idx_array.insert(doc_idx, cc_idx);
        }
    });

    let remove_count = AtomicUsize::new(0);
    let new_docs: Vec<JSONValue> = flat_with_indices 
        .into_par_iter()
        .enumerate()
        .map(|(doc_idx, (_, mut obj))| {
            if let Some(parent) = parent_lookup[doc_idx] {
                let cc_id = *cc_id_lookup.get(&parent).unwrap(); 
                let cc_size = *cc_size.get(&parent).unwrap();
                let cc_idx = *cc_idx_array.get(&doc_idx).unwrap();
                if cc_idx > 0 {
                    remove_count.fetch_add(1, Ordering::Relaxed);
                }
                json_set(&mut obj, annotate_key,
                    json!({"cc_id": cc_id, "cc_size": cc_size, "cc_idx": cc_idx}))
                    .unwrap();
            }
            obj
        }).collect();
    Ok((new_docs, remove_count.into_inner()))
}

fn write_docs(
    docs: Vec<JSONValue>,
    counter: &AtomicUsize,
    output_dir: &PathBuf,
    prefix: &str,
) -> Result<(), Error> {
    // Parallel: Serialize all docs
    let serialized: Vec<Vec<u8>> = docs
        .into_par_iter()
        .map(|doc| {
            let mut bytes = serde_json::to_vec(&doc).unwrap();
            bytes.push(b'\n');
            bytes
        })
        .collect();
    
    // Parallel: Group into chunks (just indices)
    let mut idx_groups: Vec<Vec<usize>> = Vec::new();

    let mut cur_group: Vec<usize> = Vec::new();
    let mut cur_size = 0;
    for (i,v) in serialized.iter().enumerate() {
    	let len = v.len();
    	cur_group.push(i);
    	cur_size += len;
    	if cur_size >= OUTPUT_FILE_SIZE {
            idx_groups.push(std::mem::take(&mut cur_group));
    		cur_size = 0;
    	}    	
    }
    if cur_group.len() > 0 {
    	idx_groups.push(cur_group)
    }

    idx_groups.into_par_iter().for_each(|group| {

        let output_file = output_dir.clone().join(format!(
            "{}_file_{:08}.jsonl.zst",
            prefix,
            counter.fetch_add(1, Ordering::SeqCst)));
	    let total_size: usize = group.iter().map(|&i| serialized[i].len()).sum();
	    let mut contents = Vec::with_capacity(total_size);	    
	    for &i in &group {
	        contents.extend_from_slice(&serialized[i]);
	    }        
        write_mem_to_pathbuf(&contents, &output_file).unwrap();
    });

    
    Ok(())
}

fn generate_pair_indices<T>(data: &Vec<Vec<T>>) -> Vec<(usize, usize, usize)> {
    data.iter()
        .enumerate()
        .flat_map(|(vec_idx, inner)| {
            (0..inner.len()).flat_map(move |i| ((i + 1)..inner.len()).map(move |j| (vec_idx, i, j)))
        })
        .collect()
}
