//! # Single-Machine MinHash Deduplication
//!
//! This module provides a simplified API for running the entire MinHash deduplication
//! pipeline on a single machine where all data can be processed locally.

use crate::minhash_base::{build_uf, clean_files, gather_edges, hash_only, FileMap};
use crate::minhash_config::{
    Config, ConfigOverrides, EngOverrides, MinHashOverrides, OutputOverrides,
};
use anyhow::{Error, Result};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

/// Runs the complete MinHash deduplication pipeline in a single function call.
///
/// This is a convenience function for single-machine deployments where all input data
/// can be processed locally. It sequentially executes all five pipeline stages:
///
/// 1. Build file map
/// 2. Compute MinHash signatures  
/// 3. Gather edges from matching signatures
/// 4. Build Union-Find structure
/// 5. Clean files (remove/annotate duplicates)
///
/// # Arguments
///
/// ## Required Directories
/// * `input_dir` - Directory containing `.jsonl`, `.jsonl.zst`, `.jsonl.zstd`, or `.jsonl.gz` files
/// * `storage_dir` - Working directory for intermediate files (signatures, edges, metadata)
/// * `output_dir` - Directory where deduplicated files will be written
///
/// ## Document Processing
/// * `text_key` - JSON key containing document text (e.g., "text", "content", "body")
/// * `config` - Optional path to YAML configuration file (uses defaults if None)
///
/// ## MinHash Parameters (optional overrides)
/// * `num_buckets` - Number of LSH bands (more = stricter matching, default from config)
/// * `bucket_size` - Hash values per band (more = stricter matching, default from config)
/// * `ngram_size` - Size of n-grams for shingling (default: 3 for trigrams)
/// * `permutation_seed` - Random seed for reproducibility (default from config)
/// * `tokenizer` - Tokenization strategy: "cl100k", "p50k", "uniseg", or character-level (default)
///
/// ## Output Options (optional overrides)
/// * `annotate` - If true, add duplicate metadata to documents (default: false)
/// * `annotate_key` - JSON key for annotations (default: "minhash_duplicate_info")
/// * `delete_while_cleaning` - If true, delete input files after processing (default: false)
/// * `remove_duplicates` - If true, keep only first occurrence of duplicates (default: true)
/// * `cleanup_storage` - If true, delete storage_dir after completion (default: false)
///
/// # Returns
/// `Ok(())` on success, or an error if any stage fails
///
/// # MinHash Parameters Explained
///
/// The quality/strictness of deduplication is controlled by:
/// - **num_buckets × bucket_size**: Total hash values computed per document
/// - **num_buckets**: Documents must match in at least ONE band to be considered duplicates
/// - **bucket_size**: Documents must match ALL hashes within a band
///
/// Common configurations:
/// - `num_buckets=20, bucket_size=5`: Standard (100 total hashes)
/// - `num_buckets=50, bucket_size=10`: Strict (500 total hashes, fewer false positives)
/// - `num_buckets=10, bucket_size=3`: Loose (30 total hashes, more duplicates caught)
///
/// # Output Modes
///
/// Different combinations of `annotate` and `remove_duplicates`:
///
/// | annotate | remove_duplicates | Result |
/// |----------|-------------------|--------|
/// | false    | true              | Keep first copy only, no metadata |
/// | true     | false             | Keep all copies with duplicate info |
/// | true     | true              | Keep first copy with duplicate info |
/// | false    | false             | Keep all copies, no changes |
///
/// # Annotation Format
///
/// When `annotate=true`, documents receive a JSON field:
/// ```json
/// {
///   "minhash_duplicate_info": {
///     "cc_id": 12345,      // Connected component ID
///     "cc_size": 3,        // Total duplicates in component  
///     "cc_idx": 0          // Index (0 = first, 1+ = duplicates)
///   }
/// }
/// ```
///
/// # Performance Considerations
///
/// - **Memory**: Union-Find stage requires all edges in memory. For massive datasets (billions
///   of documents), consider using the distributed API with path chunking.
/// - **Disk Space**: `storage_dir` temporarily holds:
///   - Signatures: ~(num_buckets × bucket_size × 8 bytes × num_docs)
///   - Edges: Varies based on duplicate rate
///   - Metadata: ~32 bytes per duplicate document
/// - **Time**: Roughly linear in number of documents, but Union-Find can be slow for
///   datasets with high duplication rates (many large connected components)
///
/// # Error Handling
///
/// The function will return an error if:
/// - Input directory doesn't exist or contains no valid files
/// - Storage or output directories can't be created
/// - Any pipeline stage fails (hashing, edge gathering, etc.)
/// - Cleanup fails (only if `cleanup_storage=true`)
///
/// # Notes
///
/// - All intermediate files are stored in `storage_dir` and can be deleted after completion
/// - Input files are never modified unless `delete_while_cleaning=true`
/// - Output directory structure mirrors input directory structure
/// - Progress bars and timing information are printed to stdout
/// - For multi-machine deployments, use the distributed API functions instead
///
/// # See Also
///
/// For distributed processing on multiple machines, use the individual stage functions:
/// - [`mh_build_file_map`]
/// - [`mh_hash_docs`] (parallelizable across path chunks)
/// - [`mh_gather_edges`]
/// - [`mh_build_uf`] (requires single machine)
/// - [`mh_clean_files`] (parallelizable across path chunks)
pub fn minhash_memory(
    input_dir: &PathBuf,
    storage_dir: &PathBuf,
    output_dir: &PathBuf,
    text_key: &String,
    config: &Option<PathBuf>,
    num_buckets: Option<usize>,
    bucket_size: Option<usize>,
    ngram_size: Option<usize>,
    permutation_seed: Option<u64>,
    num_sig_chunks: Option<usize>,
    tokenizer: Option<String>,
    annotate: Option<bool>,
    annotate_key: Option<String>,
    delete_while_cleaning: Option<bool>,
    remove_duplicates: Option<bool>,
    cleanup_storage: bool,
) -> Result<(), Error> {
    let start_main = Instant::now();
    println!("Starting minhash...");

    // First collect the config and resolve any overrides
    let overrides = ConfigOverrides {
        minhash_params: MinHashOverrides {
            num_buckets,
            bucket_size,
            ngram_size,
            permutation_seed,
            tokenizer,
        },
        eng_params: EngOverrides {
            num_docs: None,
            max_lines_per_path: None,
            num_sig_chunks: Some(num_sig_chunks.unwrap_or(32)),
        },
        output_params: OutputOverrides {
            annotate: annotate,
            annotate_key: annotate_key,
            delete_while_cleaning,
            remove_duplicates,
        },
    };
    let config_obj = Config::load_with_overrides(config.clone(), overrides).unwrap();

    // Build the file map to get path_ids
    let file_map = FileMap::new(input_dir, &None).unwrap();
    file_map
        .save(&storage_dir.clone().join("filemap.json.gz"))
        .unwrap();

    // Then create the hashes of all documents and store them in storage_dir
    hash_only(
        &config_obj,
        &file_map,
        storage_dir,
        text_key,
        0,
        1,
        None,
        None,
    )
    .unwrap();

    // And then group into edges and build the union find
    gather_edges(&config_obj, &file_map, storage_dir).unwrap();
    build_uf(&config_obj, &file_map, storage_dir, 1).unwrap();

    // Finally handle the cleaning of data
    clean_files(
        &config_obj,
        &file_map,
        input_dir,
        storage_dir,
        output_dir,
        0,
        1,
    )
    .unwrap();
    if cleanup_storage {
        fs::remove_dir_all(storage_dir).unwrap();
    }

    println!(
        "Finished full minhash deduplication in {:?} seconds",
        start_main.elapsed().as_secs()
    );
    Ok(())
}
