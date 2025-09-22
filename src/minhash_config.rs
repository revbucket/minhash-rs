use anyhow::{Error, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

// Constants for defaults
const DEFAULT_NUM_BUCKETS: usize = 10;
const DEFAULT_BUCKET_SIZE: usize = 20;
const DEFAULT_NGRAM_SIZE: usize = 5;
const DEFAULT_PERMUTATION_SEED: u64 = 42;
const DEFAULT_TOKENIZER: &str = "cl100k_base";
const DEFAULT_NUM_DOCS: usize = 1_000_000_000;
const DEFAULT_MAX_LINES_PER_PATH: usize = 1_000_000_000;
const DEFAULT_NUM_SIG_CHUNKS: usize = 256;
const DEFAULT_DELETE_WHILE_CLEANING: bool = false;
const DEFAULT_REMOVE_DUPLICATES: bool = true;
const DEFAULT_ANNOTATE: bool = false;
const DEFAULT_ANNOTATE_KEY: &str = "minhash.fuzzy";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub minhash_params: MinHashParams,
    pub eng_params: EngParams,
    pub output_params: OutputParams,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinHashParams {
    /// Parameters for actually HOW the minhash algorithm operates
    pub num_buckets: usize,
    pub bucket_size: usize,
    pub ngram_size: usize,
    pub permutation_seed: u64,
    pub tokenizer: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngParams {
    /// Parameters regarding some internal mechanizations
    /// In a single node setting, you probably don't need to mess with this
    /// But in the multinode setting (very large scale) you want to OVERESTIMATE num_docs/max_lines_per_path
    /// num_sig_chunks should be around ~100/TB of data
    pub num_docs: usize,
    pub max_lines_per_path: usize,
    pub num_sig_chunks: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputParams {
    /// Parameters regarding what the output should be
    /// See config guidelines readme (TODO: make config guidelines readme)
    pub annotate: bool,
    pub annotate_key: String,
    pub delete_while_cleaning: bool,
    pub remove_duplicates: bool,
}

// Override structures for CLI arguments
#[derive(Debug, Default)]
pub struct ConfigOverrides {
    pub minhash_params: MinHashOverrides,
    pub eng_params: EngOverrides,
    pub output_params: OutputOverrides,
}

#[derive(Debug, Default)]
pub struct MinHashOverrides {
    pub num_buckets: Option<usize>,
    pub bucket_size: Option<usize>,
    pub ngram_size: Option<usize>,
    pub permutation_seed: Option<u64>,
    pub tokenizer: Option<String>,
}

#[derive(Debug, Default)]
pub struct EngOverrides {
    pub num_docs: Option<usize>,
    pub max_lines_per_path: Option<usize>,
    pub num_sig_chunks: Option<usize>,
}

#[derive(Debug, Default)]
pub struct OutputOverrides {
    pub annotate: Option<bool>,
    pub annotate_key: Option<String>,
    pub delete_while_cleaning: Option<bool>,
    pub remove_duplicates: Option<bool>,
}

// Default implementations
impl Default for MinHashParams {
    fn default() -> Self {
        Self {
            num_buckets: DEFAULT_NUM_BUCKETS,
            bucket_size: DEFAULT_BUCKET_SIZE,
            ngram_size: DEFAULT_NGRAM_SIZE,
            permutation_seed: DEFAULT_PERMUTATION_SEED,
            tokenizer: DEFAULT_TOKENIZER.to_string(),
        }
    }
}

impl Default for EngParams {
    fn default() -> Self {
        Self {
            num_docs: DEFAULT_NUM_DOCS,
            max_lines_per_path: DEFAULT_MAX_LINES_PER_PATH,
            num_sig_chunks: DEFAULT_NUM_SIG_CHUNKS,
        }
    }
}

impl Default for OutputParams {
    fn default() -> Self {
        Self {
            annotate: DEFAULT_ANNOTATE,
            annotate_key: DEFAULT_ANNOTATE_KEY.to_string(),
            delete_while_cleaning: DEFAULT_DELETE_WHILE_CLEANING,
            remove_duplicates: DEFAULT_REMOVE_DUPLICATES,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            minhash_params: MinHashParams::default(),
            eng_params: EngParams::default(),
            output_params: OutputParams::default(),
        }
    }
}

// Implementation for applying overrides
impl ConfigOverrides {
    pub fn apply_to(self, config: &mut Config) {
        self.minhash_params.apply_to(&mut config.minhash_params);
        self.eng_params.apply_to(&mut config.eng_params);
        self.output_params.apply_to(&mut config.output_params);
    }
}

impl MinHashOverrides {
    fn apply_to(self, config: &mut MinHashParams) {
        if let Some(val) = self.num_buckets {
            config.num_buckets = val;
        }
        if let Some(val) = self.bucket_size {
            config.bucket_size = val;
        }
        if let Some(val) = self.ngram_size {
            config.ngram_size = val;
        }
        if let Some(val) = self.permutation_seed {
            config.permutation_seed = val;
        }
        if let Some(val) = self.tokenizer {
            config.tokenizer = val;
        }
    }
}

impl EngOverrides {
    fn apply_to(self, config: &mut EngParams) {
        if let Some(val) = self.num_docs {
            config.num_docs = val;
        }
        if let Some(val) = self.max_lines_per_path {
            config.max_lines_per_path = val;
        }
        if let Some(val) = self.num_sig_chunks {
            config.num_sig_chunks = val;
        }
    }
}

impl OutputOverrides {
    fn apply_to(self, config: &mut OutputParams) {
        if let Some(val) = self.annotate_key {
            config.annotate_key = val;
        }
        if let Some(val) = self.delete_while_cleaning {
            config.delete_while_cleaning = val;
        }
        if let Some(val) = self.remove_duplicates {
            config.remove_duplicates = val;
        }
    }
}

// Main config loading logic
impl Config {
    pub fn load_with_overrides(
        config_path: Option<PathBuf>,
        overrides: ConfigOverrides,
    ) -> Result<Self, Error> {
        // Start with defaults
        let mut config = Self::default();

        // Apply config file if present
        if let Some(path) = config_path {
            let config_content = fs::read_to_string(path)?;
            let file_config: Config = serde_yaml::from_str(&config_content)?;
            config = file_config;
        }

        // Apply CLI overrides
        overrides.apply_to(&mut config);

        Ok(config)
    }

    /// Convenience method for loading from just a file path
    pub fn load_from_file(config_path: PathBuf) -> Result<Self, Error> {
        Self::load_with_overrides(Some(config_path), ConfigOverrides::default())
    }

    /// Create config with only CLI overrides (no file)
    pub fn from_overrides(overrides: ConfigOverrides) -> Self {
        let mut config = Self::default();
        overrides.apply_to(&mut config);
        config
    }
}
