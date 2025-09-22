use crate::minhash_base::{build_uf, clean_files, gather_edges, hash_only, FileMap};
use crate::minhash_config::{
    Config, ConfigOverrides, EngOverrides, MinHashOverrides, OutputOverrides,
};
use anyhow::{Error, Result};
use std::fs;
use std::path::PathBuf;

fn get_file_map_loc(storage_dir: &PathBuf) -> PathBuf {
    storage_dir.clone().join("filemap.json.gz")
}

pub fn mh_build_file_map(input_dir: &PathBuf, storage_dir: &PathBuf) -> Result<(), Error> {
    let file_map = FileMap::new(input_dir, &None).unwrap();
    let file_map_loc = get_file_map_loc(storage_dir);
    file_map.save(&file_map_loc)
}

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
        Some(local_input.clone())
    )
}

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
