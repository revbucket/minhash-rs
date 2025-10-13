//! # MinHash Deduplication API
//!
//! This module provides the main entry points for running the MinHash deduplication pipeline.
//! Each function corresponds to one stage of the pipeline and can be called independently
//! to support distributed processing across multiple machines or path chunks.
//!
//! ## Pipeline Stages
//!
//! 1. **Build File Map** (`mh_build_file_map`): Create index of all files
//! 2. **Hash Documents** (`mh_hash_docs`): Compute MinHash signatures
//! 3. **Gather Edges** (`mh_gather_edges`): Collect duplicate candidates
//! 4. **Build Union-Find** (`mh_build_uf`): Identify connected components
//! 5. **Clean Files** (`mh_clean_files`): Apply deduplication

use crate::minhash_base::{build_uf, clean_files, gather_edges, hash_only, FileMap};
use crate::minhash_config::{
    Config, ConfigOverrides, EngOverrides, MinHashOverrides, OutputOverrides,
};
use anyhow::{Error, Result};
use std::fs;
use std::path::PathBuf;

/// Returns the standard location for the file map within the storage directory.
fn get_file_map_loc(storage_dir: &PathBuf) -> PathBuf {
    storage_dir.clone().join("filemap.json.gz")
}

/// **Stage 1**: Builds and saves a file map for all input files.
///
/// This function scans the input directory and creates a mapping from file paths
/// to integer indices. The mapping is saved to `storage_dir/filemap.json.gz` and
/// will be loaded by subsequent pipeline stages.
///
/// # Arguments
/// * `input_dir` - Directory containing `.jsonl`, `.jsonl.zst`, `.jsonl.zstd`, or `.jsonl.gz` files
/// * `storage_dir` - Directory where the file map will be saved
///
/// # Returns
/// `Ok(())` on success, or an error if directory scanning or file writing fails
///
/// # Example
/// ```no_run
/// use std::path::PathBuf;
///
/// let input_dir = PathBuf::from("/data/documents");
/// let storage_dir = PathBuf::from("/scratch/minhash_storage");
///
/// mh_build_file_map(&input_dir, &storage_dir)?;
/// ```
///
/// # Notes
/// - Must be run before any other pipeline stages
/// - For multi-node setups with remote storage, use the Python tool instead
pub fn mh_build_file_map(input_dir: &PathBuf, storage_dir: &PathBuf) -> Result<(), Error> {
    let file_map = FileMap::new(input_dir, &None).unwrap();
    let file_map_loc = get_file_map_loc(storage_dir);
    file_map.save(&file_map_loc)
}

/// **Stage 2**: Computes MinHash signatures for a chunk of documents.
///
/// This function processes a subset of input files (determined by `path_chunk` and
/// `num_path_chunks`) and generates MinHash signatures for all documents. Signatures
/// are stored in `storage_dir/sig_storage/` organized by band and signature chunk.
///
/// # Arguments
/// * `local_input` - Local directory containing input files to process
/// * `storage_dir` - Directory containing file map and where signatures will be saved
/// * `text_key` - JSON key containing the document text (e.g., "text", "content")
/// * `config` - Optional path to YAML configuration file
/// * `path_chunk` - Zero-indexed chunk ID for this worker (0 to num_path_chunks-1)
/// * `num_path_chunks` - Total number of chunks for parallel processing
/// * `num_buckets` - Override: Number of LSH bands (default from config)
/// * `bucket_size` - Override: Number of hash values per band (default from config)
/// * `ngram_size` - Override: Size of n-grams for shingling (default from config)
/// * `permutation_seed` - Override: Random seed for hash permutations (default from config)
/// * `tokenizer` - Override: Tokenizer name ("cl100k", "p50k", "uniseg", or character-level)
/// * `num_docs` - Override: Expected total number of documents (affects signature size)
/// * `max_lines_per_path` - Override: Maximum lines per file (affects encoding size)
/// * `num_sig_chunks` - Override: Number of signature chunks for distribution
///
/// # Returns
/// `Ok(())` on success, or an error if processing fails
///
/// # Parallelism
/// Multiple workers can run this function concurrently with different `path_chunk` values.
/// Each worker processes an independent subset of files.
///
/// # Example
/// ```no_run
/// // Worker processing chunk 0 of 4
/// mh_hash_docs(
///     &PathBuf::from("/data/documents"),
///     &PathBuf::from("/scratch/storage"),
///     &"text".to_string(),
///     &None,  // Use default config
///     0,      // path_chunk
///     4,      // num_path_chunks
///     Some(20),   // 20 bands
///     Some(5),    // 5 hashes per band
///     Some(3),    // trigrams
///     None, None, None, None, None
/// )?;
/// ```
pub fn mh_hash_docs(
    local_input: &PathBuf,
    storage_dir: &PathBuf,
    text_key: &String,
    config: &Option<PathBuf>,
    path_chunk: usize,
    num_path_chunks: usize,
    num_buckets: Option<usize>,
    bucket_size: Option<usize>,
    ngram_size: Option<usize>,
    permutation_seed: Option<u64>,
    tokenizer: Option<String>,
    num_docs: Option<usize>,
    max_lines_per_path: Option<usize>,
    num_sig_chunks: Option<usize>,
) -> Result<(), Error> {
    let overrides = ConfigOverrides {
        minhash_params: MinHashOverrides {
            num_buckets,
            bucket_size,
            ngram_size,
            permutation_seed,
            tokenizer,
        },
        eng_params: EngOverrides {
            num_docs: num_docs,
            max_lines_per_path: max_lines_per_path,
            num_sig_chunks: num_sig_chunks,
        },
        output_params: OutputOverrides::default(),
    };
    let config_obj = Config::load_with_overrides(config.clone(), overrides).unwrap();

    let file_map = FileMap::load(&get_file_map_loc(storage_dir)).unwrap();

    hash_only(
        &config_obj,
        &file_map,
        storage_dir,
        text_key,
        path_chunk,
        num_path_chunks,
        None,
        Some(local_input.clone()),
    )
}

/// **Stage 3**: Gathers edges by grouping documents with matching signatures.
///
/// This function reads all signature files and creates edge files containing pairs
/// of (path_id, line_num) for documents that share the same signature in at least
/// one LSH band. Edge files are saved to `storage_dir/edges/`.
///
/// # Arguments
/// * `storage_dir` - Directory containing signature files and where edges will be saved
/// * `config` - Optional path to YAML configuration file
/// * `num_docs` - Override: Expected total number of documents
/// * `max_lines_per_path` - Override: Maximum lines per file
///
/// # Returns
/// `Ok(())` on success, or an error if processing fails
///
/// # Parallelism
/// This stage parallelizes across bands internally. All signature files must be
/// accessible from the machine running this function.
///
/// # Notes
/// - Requires all signature files from Stage 2 to be present
/// - In multi-node setups, gather signature files for each band onto a single machine first
pub fn mh_gather_edges(
    storage_dir: &PathBuf,
    config: &Option<PathBuf>,
    num_docs: Option<usize>,
    max_lines_per_path: Option<usize>,
) -> Result<(), Error> {
    let overrides = ConfigOverrides {
        minhash_params: MinHashOverrides::default(),
        eng_params: EngOverrides {
            num_docs: num_docs,
            max_lines_per_path: max_lines_per_path,
            num_sig_chunks: None,
        },
        output_params: OutputOverrides::default(),
    };
    let config_obj = Config::load_with_overrides(config.clone(), overrides).unwrap();

    let file_map = FileMap::load(&get_file_map_loc(storage_dir)).unwrap();

    gather_edges(&config_obj, &file_map, storage_dir)
}

/// **Stage 4**: Builds a Union-Find structure to identify connected components of duplicates.
///
/// This function processes all edge files and uses a Union-Find algorithm to merge
/// documents into connected components. It generates cleaning metadata files that
/// specify which documents are duplicates and which copy to keep.
///
/// Metadata files are saved to `storage_dir/clean/chunk_*.clean.bin`.
///
/// # Arguments
/// * `storage_dir` - Directory containing edge files and where cleaning metadata will be saved
/// * `config` - Optional path to YAML configuration file
/// * `num_path_chunks` - Number of chunks for partitioning cleaning metadata
/// * `max_lines_per_path` - Override: Maximum lines per file
///
/// # Returns
/// `Ok(())` on success, or an error if processing fails
///
/// # Parallelism
/// **WARNING**: This stage does NOT support multi-node parallelism. It must be run
/// on a single machine with access to all edge files. Internal operations are
/// parallelized across CPU cores.
///
/// # Notes
/// - Most memory-intensive stage of the pipeline
/// - Requires all edge files from Stage 3 to be present
/// - `num_path_chunks` should match the value used in Stage 5
pub fn mh_build_uf(
    storage_dir: &PathBuf,
    config: Option<PathBuf>,
    num_path_chunks: usize,
    max_lines_per_path: Option<usize>,
) -> Result<(), Error> {
    let overrides = ConfigOverrides {
        minhash_params: MinHashOverrides::default(),
        eng_params: EngOverrides {
            num_docs: None,
            max_lines_per_path: max_lines_per_path,
            num_sig_chunks: None,
        },
        output_params: OutputOverrides::default(),
    };
    let config_obj = Config::load_with_overrides(config.clone(), overrides).unwrap();

    let file_map = FileMap::load(&get_file_map_loc(storage_dir)).unwrap();

    build_uf(&config_obj, &file_map, storage_dir, num_path_chunks)
}

/// **Stage 5**: Applies deduplication by cleaning input files.
///
/// This function reads the cleaning metadata from Stage 4 and processes input files
/// to either remove duplicates, annotate documents with duplicate information, or both.
/// Cleaned files are written to `output_dir`.
///
/// # Arguments
/// * `input_dir` - Directory containing original input files
/// * `storage_dir` - Directory containing cleaning metadata from Stage 4
/// * `output_dir` - Directory where cleaned files will be written
/// * `path_chunk` - Zero-indexed chunk ID for this worker
/// * `num_path_chunks` - Total number of chunks (must match Stage 4)
/// * `config` - Optional path to YAML configuration file
/// * `annotate` - If true, add duplicate metadata to documents as JSON
/// * `annotate_key` - JSON key for annotation (e.g., "duplicate_info")
/// * `delete_while_cleaning` - If true, delete input files after processing
/// * `remove_duplicates` - If true, keep only first occurrence of each duplicate
/// * `cleanup_storage` - If true, delete entire storage directory after completion
///
/// # Returns
/// `Ok(())` on success, or an error if processing fails
///
/// # Annotation Format
/// When `annotate=true`, each document receives a JSON field with:
/// ```json
/// {
///   "cc_id": 12345,        // Connected component ID
///   "cc_size": 3,          // Total duplicates in component
///   "cc_idx": 0            // Index in component (0 = first occurrence)
/// }
/// ```
///
/// # Parallelism
/// Multiple workers can run this function concurrently with different `path_chunk` values.
///
/// # Example
/// ```no_run
/// // Worker 0 of 4: Remove duplicates and annotate survivors
/// mh_clean_files(
///     &PathBuf::from("/data/documents"),
///     &PathBuf::from("/scratch/storage"),
///     &PathBuf::from("/data/deduplicated"),
///     0,      // path_chunk
///     4,      // num_path_chunks
///     &None,  // Use default config
///     Some(true),                    // Annotate
///     Some("dedup_info".to_string()), // Annotation key
///     Some(false),                   // Don't delete originals
///     Some(true),                    // Remove duplicates
///     false                          // Keep storage dir
/// )?;
/// ```
///
/// # Notes
/// - Set `cleanup_storage=true` only on the final worker to avoid race conditions
/// - If only annotating (not removing), all documents appear in output with metadata
/// - Deduplication preserves the first occurrence of each duplicate cluster
pub fn mh_clean_files(
    input_dir: &PathBuf,
    storage_dir: &PathBuf,
    output_dir: &PathBuf,
    path_chunk: usize,
    num_path_chunks: usize,
    config: &Option<PathBuf>,
    annotate: Option<bool>,
    annotate_key: Option<String>,
    delete_while_cleaning: Option<bool>,
    remove_duplicates: Option<bool>,
    cleanup_storage: bool,
) -> Result<(), Error> {
    let overrides = ConfigOverrides {
        minhash_params: MinHashOverrides::default(),
        eng_params: EngOverrides {
            num_docs: None,
            max_lines_per_path: None,
            num_sig_chunks: None,
        },
        output_params: OutputOverrides {
            annotate: annotate,
            annotate_key: annotate_key,
            delete_while_cleaning: delete_while_cleaning,
            remove_duplicates: remove_duplicates,
        },
    };
    let config_obj = Config::load_with_overrides(config.clone(), overrides).unwrap();

    let file_map = FileMap::load(&get_file_map_loc(storage_dir)).unwrap();
    clean_files(
        &config_obj,
        &file_map,
        input_dir,
        storage_dir,
        output_dir,
        path_chunk,
        num_path_chunks,
    )
    .unwrap();
    if cleanup_storage {
        fs::remove_dir_all(storage_dir).unwrap();
    }

    Ok(())
}
