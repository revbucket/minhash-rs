use crate::minhash_base::{build_uf, clean_files, gather_edges, hash_only, FileMap};
use crate::minhash_config::{
    Config, ConfigOverrides, EngOverrides, MinHashOverrides, OutputOverrides,
};
use anyhow::{Error, Result};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
/*

*/

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
            num_sig_chunks: None,
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

    // Then create the hashes of all documents and store them in storage_dir
    hash_only(&config_obj, &file_map, storage_dir, text_key, 0, 1, None, None).unwrap();

    // And then group into edges and build the union find
    gather_edges(&config_obj, &file_map, storage_dir).unwrap();
    build_uf(&config_obj, &file_map, storage_dir, 0).unwrap();

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
