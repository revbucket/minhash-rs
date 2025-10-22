//! # MinHash Component Analysis
//!
//! This module analyzes the structure and quality of connected components produced by MinHash LSH.
//! It computes true Jaccard similarities for all pairs within components and classifies edges as:
//! - **Direct edges**: Pairs that matched via MinHash band collision
//! - **Transitive edges**: Pairs grouped together via Union-Find but without direct band match
//!
//! This enables answering questions like:
//! - How many components are fully connected (all pairs matched directly)?
//! - What is the false positive rate (direct edges with low true Jaccard)?
//! - What is the false negative rate (missing edges with high true Jaccard)?

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{BuildHasher, Hash, Hasher};
use std::io::{BufRead, BufWriter, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use std::fs::File;

use ahash::RandomState;
use anyhow::Result;
use dashmap::DashMap;
use mj_io::{build_pbar, expand_dirs, read_pathbuf_to_mem};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value as JSONValue;

use crate::minhash_base::{preprocess_text, FileMap, OmniTokenizer};
use crate::minhash_config::{Config, ConfigOverrides, EngOverrides, MinHashOverrides, OutputOverrides};
use crate::storage::{IntValueEnum, to_byte_size};

/// Statistics for a single edge within a component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeInfo {
    pub doc1_id: String,
    pub doc2_id: String,
    pub jaccard: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bands: Option<Vec<usize>>,
}

/// Analysis results for a single connected component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentAnalysis {
    pub cc_id: usize,
    pub cc_size: usize,
    pub num_direct_edges: usize,
    pub num_transitive_edges: usize,
    pub density: f64,
    pub is_fully_connected: bool,
    pub direct_edges: Vec<EdgeInfo>,
    pub transitive_edges: Vec<EdgeInfo>,
    pub stats: ComponentStats,
}

/// Statistical summary for component edges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentStats {
    pub direct_jaccard_mean: f64,
    pub direct_jaccard_min: f64,
    pub direct_jaccard_max: f64,
    pub direct_jaccard_std: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transitive_jaccard_mean: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transitive_jaccard_min: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transitive_jaccard_max: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transitive_jaccard_std: Option<f64>,
}

/// Main entry point for component analysis
pub fn analyze_components(
    input_dir: &PathBuf,
    storage_dir: &PathBuf,
    output_file: &PathBuf,
    text_key: &str,
    config: Option<PathBuf>,
) -> Result<()> {
    let start_main = Instant::now();
    println!("Starting MinHash component analysis");

    // Load config
    let overrides = ConfigOverrides {
        minhash_params: MinHashOverrides {
            num_buckets: None,
            bucket_size: None,
            ngram_size: None,
            permutation_seed: None,
            tokenizer: None,
        },
        eng_params: EngOverrides {
            num_docs: None,
            max_lines_per_path: None,
            num_sig_chunks: None,
        },
        output_params: OutputOverrides {
            annotate: None,
            annotate_key: None,
            delete_while_cleaning: None,
            remove_duplicates: None,
        },
    };
    let config_obj = Config::load_with_overrides(config, overrides)?;
    let ngram_size = config_obj.minhash_params.ngram_size;
    let tokenizer_name = config_obj.minhash_params.tokenizer;
    let tokenizer = OmniTokenizer::new(&tokenizer_name)?;

    // Step 1: Load MinHash results
    println!("Loading MinHash results...");
    let file_map = FileMap::load(&storage_dir.join("filemap.json.gz"))?;
    let direct_edges = load_minhash_edges(storage_dir, &config_obj.eng_params, &file_map)?;
    let components = load_components(storage_dir, &config_obj.eng_params, &file_map)?;

    println!("Found {} components with {} total direct edges",
             components.len(), direct_edges.len());

    // Step 2: Load all documents into memory
    println!("Loading documents...");
    let documents = load_documents(input_dir, &file_map, &config_obj.eng_params)?;
    println!("Loaded {} documents", documents.len());

    // Step 3: Analyze each component
    println!("Analyzing components...");
    let output_writer = Arc::new(std::sync::Mutex::new(BufWriter::new(File::create(output_file)?)));
    let pbar = build_pbar(components.len(), "Components");

    components.par_iter().for_each(|(cc_id, doc_indices)| {
        if let Ok(analysis) = analyze_single_component(
            *cc_id,
            doc_indices,
            &direct_edges,
            &documents,
            &file_map,
            &tokenizer,
            ngram_size,
            text_key,
        ) {
            let json_str = serde_json::to_string(&analysis).unwrap();
            let mut writer = output_writer.lock().unwrap();
            writeln!(writer, "{}", json_str).unwrap();
        }
        pbar.inc(1);
    });

    let mut writer = output_writer.lock().unwrap();
    writer.flush()?;

    println!("Finished component analysis in {} secs", start_main.elapsed().as_secs());
    Ok(())
}

/// Load all documents from input directory using MinHash indexing scheme
/// Documents are indexed as: doc_idx = path_id * max_lines_per_path + line_num
fn load_documents(
    input_dir: &PathBuf,
    file_map: &FileMap,
    eng_config: &crate::minhash_config::EngParams,
) -> Result<HashMap<usize, JSONValue>> {
    let docs: DashMap<usize, JSONValue> = DashMap::new();

    // Iterate through files in FileMap to match MinHash indexing
    file_map.indices.par_iter().for_each(|(path, &path_id)| {
        if let Ok(contents) = read_pathbuf_to_mem(&input_dir.join(path)) {
            for (line_num, line_result) in contents.lines().enumerate() {
                if let Ok(line) = line_result {
                    if let Ok(doc) = serde_json::from_str::<JSONValue>(&line) {
                        let doc_idx = path_id * eng_config.max_lines_per_path + line_num;
                        docs.insert(doc_idx, doc.clone());
                    }
                }
            }
        }
    });

    Ok(docs.into_iter().collect())
}

/// Load MinHash direct edges from storage directory
fn load_minhash_edges(
    storage_dir: &PathBuf,
    eng_config: &crate::minhash_config::EngParams,
    file_map: &FileMap,
) -> Result<HashMap<(usize, usize), Vec<usize>>> {
    println!("Loading MinHash edges from storage...");
    let path_size = to_byte_size(file_map.indices.len());
    let line_size = to_byte_size(eng_config.max_lines_per_path);
    let entry_size = path_size + line_size;

    // Recursively find all edge files in subdirectories
    let edges_dir = storage_dir.join("edges");
    let sigchunk_dirs = std::fs::read_dir(&edges_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.path())
        .collect::<Vec<_>>();

    let edge_files = expand_dirs(
        sigchunk_dirs,
        Some(vec![".edges.bin"].as_slice()),
    )?;

    println!("Found {} edge files to process", edge_files.len());

    let edges: DashMap<(usize, usize), Vec<usize>> = DashMap::new();
    let pbar = build_pbar(edge_files.len(), "Edge files");

    edge_files.par_iter().for_each(|edge_file| {
        // Extract band ID from filename: "band_{:08}.edges.bin"
        let band_id = edge_file
            .file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.strip_prefix("band_"))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);

        let contents = read_pathbuf_to_mem(edge_file).unwrap().into_inner().into_inner();

        // Create sentinel values - note: uses (1 << size - 1) NOT (1 << size) - 1
        let max_path = IntValueEnum::new(1 << path_size - 1, path_size);
        let max_line = IntValueEnum::new(1 << line_size - 1, line_size);
        let sentinel_path = max_path.as_uint::<usize>();
        let sentinel_line = max_line.as_uint::<usize>();

        let mut current_clique: Vec<(usize, usize)> = Vec::new();

        // Use chunks_exact to ensure all chunks are exactly entry_size
        for chunk in contents.chunks_exact(entry_size) {
            let path_id = IntValueEnum::from_bytes(chunk[..path_size].to_vec(), path_size)
                .as_uint::<usize>();
            let line_id = IntValueEnum::from_bytes(chunk[path_size..].to_vec(), line_size)
                .as_uint::<usize>();

            if path_id == sentinel_path && line_id == sentinel_line {
                // End of clique marker
                if current_clique.len() >= 2 {
                    // Add all pairs in this clique
                    for i in 0..current_clique.len() {
                        for j in (i + 1)..current_clique.len() {
                            let (path_i, line_i) = current_clique[i];
                            let (path_j, line_j) = current_clique[j];
                            let doc_i = path_i * eng_config.max_lines_per_path + line_i;
                            let doc_j = path_j * eng_config.max_lines_per_path + line_j;
                            let key = if doc_i < doc_j {
                                (doc_i, doc_j)
                            } else {
                                (doc_j, doc_i)
                            };
                            edges.entry(key).or_default().push(band_id);
                        }
                    }
                }
                current_clique.clear();
            } else {
                current_clique.push((path_id, line_id));
            }
        }
        pbar.inc(1);
    });

    Ok(edges.into_iter().collect())
}

/// Load connected components from UF metadata
fn load_components(
    storage_dir: &PathBuf,
    eng_config: &crate::minhash_config::EngParams,
    file_map: &FileMap,
) -> Result<HashMap<usize, Vec<usize>>> {
    println!("Loading component structure from clean metadata...");
    let path_size = to_byte_size(file_map.indices.len());
    let line_size = to_byte_size(eng_config.max_lines_per_path);

    let metadata_files = expand_dirs(
        vec![storage_dir.join("clean")],
        Some(vec![".clean.bin"].as_slice()),
    )?;

    let components: DashMap<usize, Vec<usize>> = DashMap::new();
    let pbar = build_pbar(metadata_files.len(), "Clean files");

    metadata_files.par_iter().for_each(|meta_file| {
        let contents = read_pathbuf_to_mem(meta_file).unwrap().into_inner().into_inner();

        // Parse header: [path_size, line_size, cc_id_size, cc_size_byte_size, cc_size_byte_size]
        let header_path_size = u64::from_le_bytes(contents[0..8].try_into().unwrap()) as usize;
        let header_line_size = u64::from_le_bytes(contents[8..16].try_into().unwrap()) as usize;
        let cc_id_size = u64::from_le_bytes(contents[16..24].try_into().unwrap()) as usize;
        let cc_size_size = u64::from_le_bytes(contents[24..32].try_into().unwrap()) as usize;
        let cc_idx_size = u64::from_le_bytes(contents[32..40].try_into().unwrap()) as usize;

        let entry_size = header_path_size + header_line_size + cc_id_size + cc_size_size + cc_idx_size;

        // Skip 40-byte header and parse entries
        for chunk in contents[40..].chunks(entry_size) {
            if chunk.len() < entry_size {
                break;
            }

            let path_id = IntValueEnum::from_bytes(chunk[..header_path_size].to_vec(), header_path_size)
                .as_uint::<usize>();
            let line_id = IntValueEnum::from_bytes(chunk[header_path_size..header_path_size + header_line_size].to_vec(), header_line_size)
                .as_uint::<usize>();
            let cc_id = IntValueEnum::from_bytes(chunk[header_path_size + header_line_size..header_path_size + header_line_size + cc_id_size].to_vec(), cc_id_size)
                .as_uint::<usize>();

            let doc_idx = path_id * eng_config.max_lines_per_path + line_id;
            components.entry(cc_id).or_default().push(doc_idx);
        }
        pbar.inc(1);
    });

    Ok(components.into_iter().collect())
}

/// Analyze a single component
fn analyze_single_component(
    cc_id: usize,
    doc_indices: &[usize],
    direct_edges: &HashMap<(usize, usize), Vec<usize>>,
    documents: &HashMap<usize, JSONValue>,
    file_map: &FileMap,
    tokenizer: &OmniTokenizer,
    ngram_size: usize,
    text_key: &str,
) -> Result<ComponentAnalysis> {
    let cc_size = doc_indices.len();

    // Compute all pairwise Jaccard similarities
    let jaccard_results = compute_component_jaccard(doc_indices, documents, file_map, tokenizer, ngram_size, text_key)?;

    // Classify edges as direct or transitive
    let mut direct_edge_infos = Vec::new();
    let mut transitive_edge_infos = Vec::new();

    for (i, &doc_i) in doc_indices.iter().enumerate() {
        for &doc_j in &doc_indices[i + 1..] {
            let key = if doc_i < doc_j {
                (doc_i, doc_j)
            } else {
                (doc_j, doc_i)
            };

            if let Some(jaccard) = jaccard_results.get(&key) {
                let doc_i_id = format!("doc_{}", doc_i);
                let doc_j_id = format!("doc_{}", doc_j);

                let edge_info = EdgeInfo {
                    doc1_id: doc_i_id.clone(),
                    doc2_id: doc_j_id.clone(),
                    jaccard: *jaccard,
                    bands: direct_edges.get(&key).cloned(),
                };

                if direct_edges.contains_key(&key) {
                    direct_edge_infos.push(edge_info);
                } else {
                    transitive_edge_infos.push(edge_info);
                }
            }
        }
    }

    // Compute statistics
    let total_possible_edges = (cc_size * (cc_size - 1)) / 2;
    let num_direct = direct_edge_infos.len();
    let num_transitive = transitive_edge_infos.len();
    let density = num_direct as f64 / total_possible_edges as f64;
    let is_fully_connected = num_direct == total_possible_edges;

    let stats = compute_stats(&direct_edge_infos, &transitive_edge_infos);

    Ok(ComponentAnalysis {
        cc_id,
        cc_size,
        num_direct_edges: num_direct,
        num_transitive_edges: num_transitive,
        density,
        is_fully_connected,
        direct_edges: direct_edge_infos,
        transitive_edges: transitive_edge_infos,
        stats,
    })
}

/// Compute pairwise Jaccard for all documents in a component
fn compute_component_jaccard(
    doc_indices: &[usize],
    documents: &HashMap<usize, JSONValue>,
    _file_map: &FileMap,
    tokenizer: &OmniTokenizer,
    ngram_size: usize,
    text_key: &str,
) -> Result<HashMap<(usize, usize), f64>> {
    // First, create token sets for all documents
    let toksets: HashMap<usize, HashSet<u64>> = doc_indices
        .iter()
        .filter_map(|&doc_idx| {
            let doc = documents.get(&doc_idx)?;
            let text = doc.get(text_key)?.as_str()?.to_string();
            let tokset = text_2_tokset(&text, tokenizer, ngram_size).ok()?;
            Some((doc_idx, tokset))
        })
        .collect();

    // Now compute Jaccard for all pairs
    let mut results = HashMap::new();
    for i in 0..doc_indices.len() {
        for j in (i + 1)..doc_indices.len() {
            let doc_i = doc_indices[i];
            let doc_j = doc_indices[j];

            if let (Some(set_i), Some(set_j)) = (toksets.get(&doc_i), toksets.get(&doc_j)) {
                let jaccard = compute_jaccard(set_i, set_j);
                let key = if doc_i < doc_j {
                    (doc_i, doc_j)
                } else {
                    (doc_j, doc_i)
                };
                results.insert(key, jaccard);
            }
        }
    }

    Ok(results)
}

/// Convert text to token ngram set (reused from true_jaccard.rs)
fn text_2_tokset(
    text: &str,
    tokenizer: &OmniTokenizer,
    ngram_size: usize,
) -> Result<HashSet<u64>> {
    let mut tokset: HashSet<u64> = HashSet::new();
    let tokens = preprocess_text(text, tokenizer);

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

/// Compute Jaccard similarity between two sets
fn compute_jaccard(set1: &HashSet<u64>, set2: &HashSet<u64>) -> f64 {
    if set1.is_empty() && set2.is_empty() {
        return 1.0;
    }

    let intersection: usize = set1.iter().filter(|x| set2.contains(x)).count();
    let union = set1.len() + set2.len() - intersection;

    if union == 0 {
        return 0.0;
    }

    intersection as f64 / union as f64
}

/// Compute statistical summary for edges
fn compute_stats(direct_edges: &[EdgeInfo], transitive_edges: &[EdgeInfo]) -> ComponentStats {
    let direct_jaccards: Vec<f64> = direct_edges.iter().map(|e| e.jaccard).collect();

    let (direct_mean, direct_std) = mean_std(&direct_jaccards);
    let direct_min = if direct_jaccards.is_empty() {
        0.0
    } else {
        direct_jaccards.iter().cloned().fold(f64::INFINITY, f64::min)
    };
    let direct_max = if direct_jaccards.is_empty() {
        0.0
    } else {
        direct_jaccards.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    };

    let (trans_mean, trans_min, trans_max, trans_std) = if transitive_edges.is_empty() {
        (None, None, None, None)
    } else {
        let trans_jaccards: Vec<f64> = transitive_edges.iter().map(|e| e.jaccard).collect();
        let (mean, std) = mean_std(&trans_jaccards);
        let min = trans_jaccards.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = trans_jaccards.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (Some(mean), Some(min), Some(max), Some(std))
    };

    ComponentStats {
        direct_jaccard_mean: direct_mean,
        direct_jaccard_min: direct_min,
        direct_jaccard_max: direct_max,
        direct_jaccard_std: direct_std,
        transitive_jaccard_mean: trans_mean,
        transitive_jaccard_min: trans_min,
        transitive_jaccard_max: trans_max,
        transitive_jaccard_std: trans_std,
    }
}

/// Compute mean and standard deviation
fn mean_std(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }

    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
    let std = variance.sqrt();

    (mean, std)
}
