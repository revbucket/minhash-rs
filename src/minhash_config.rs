//! # MinHash Configuration System
//!
//! Configuration management for MinHash deduplication with support for YAML files,
//! defaults, and command-line overrides.
//!
//! ## Configuration Hierarchy
//!
//! Configuration values are resolved in the following priority order (highest to lowest):
//! 1. **CLI overrides**: Command-line arguments (highest priority)
//! 2. **YAML config file**: Values from a specified configuration file
//! 3. **Defaults**: Built-in default values (lowest priority)
//!
//! ## Configuration Groups
//!
//! Settings are organized into three categories:
//!
//! ### MinHash Parameters
//! Controls the core deduplication algorithm behavior:
//! - `num_buckets`: Number of LSH bands (default: 10)
//! - `bucket_size`: Hash values per band (default: 20)
//! - `ngram_size`: N-gram size for shingling (default: 5)
//! - `permutation_seed`: Random seed for reproducibility (default: 42)
//! - `tokenizer`: Tokenization strategy (default: "cl100k_base")
//!
//! ### Engineering Parameters
//! Internal settings for distributed processing:
//! - `num_docs`: Expected total documents (default: 1 billion)
//! - `max_lines_per_path`: Maximum lines per file (default: 1 billion)
//! - `num_sig_chunks`: Signature partitions (default: 256)
//!
//! ### Output Parameters
//! Controls output behavior:
//! - `annotate`: Add duplicate metadata (default: false)
//! - `annotate_key`: JSON key for annotations (default: "minhash.fuzzy")
//! - `delete_while_cleaning`: Delete input files (default: false)
//! - `remove_duplicates`: Remove duplicates vs. annotate only (default: true)
//!
//! ## Example YAML Configuration
//!
//! ```yaml
//! minhash_params:
//!   num_buckets: 20
//!   bucket_size: 5
//!   ngram_size: 3
//!   permutation_seed: 12345
//!   tokenizer: "cl100k_base"
//!
//! eng_params:
//!   num_docs: 10000000000  # 10 billion
//!   max_lines_per_path: 5000000
//!   num_sig_chunks: 500
//!
//! output_params:
//!   annotate: true
//!   annotate_key: "duplicate_info"
//!   delete_while_cleaning: false
//!   remove_duplicates: true
//! ```
//!
//! ## Usage Examples
//!
//! ```no_run
//! use std::path::PathBuf;
//!
//! // Load from file only
//! let config = Config::load_from_file(PathBuf::from("config.yaml"))?;
//!
//! // Use defaults only
//! let config = Config::default();
//!
//! // Load file with CLI overrides
//! let overrides = ConfigOverrides {
//!     minhash_params: MinHashOverrides {
//!         num_buckets: Some(50),
//!         bucket_size: Some(10),
//!         ..Default::default()
//!     },
//!     ..Default::default()
//! };
//! let config = Config::load_with_overrides(
//!     Some(PathBuf::from("config.yaml")),
//!     overrides
//! )?;
//!
//! // CLI overrides only (no file)
//! let config = Config::from_overrides(overrides);
//! ```

use anyhow::{Error, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

// Default constants
const DEFAULT_NUM_BUCKETS: usize = 14;
const DEFAULT_BUCKET_SIZE: usize = 8;
const DEFAULT_NGRAM_SIZE: usize = 5;
const DEFAULT_PERMUTATION_SEED: u64 = 42;
const DEFAULT_TOKENIZER: &str = "cl100k_base";
const DEFAULT_NUM_DOCS: usize = 1_000_000_000;
const DEFAULT_MAX_LINES_PER_PATH: usize = 1_000_000_000;
const DEFAULT_NUM_SIG_CHUNKS: usize = 128;
const DEFAULT_DELETE_WHILE_CLEANING: bool = false;
const DEFAULT_REMOVE_DUPLICATES: bool = true;
const DEFAULT_ANNOTATE: bool = false;
const DEFAULT_ANNOTATE_KEY: &str = "minhash.fuzzy";

/// Complete configuration for MinHash deduplication.
///
/// Contains all parameters needed to run the deduplication pipeline,
/// organized into three logical groups.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub minhash_params: MinHashParams,
    pub eng_params: EngParams,
    pub output_params: OutputParams,
}

/// Core MinHash algorithm parameters.
///
/// These settings control the quality and behavior of duplicate detection.
///
/// ## Parameter Guidelines
///
/// ### num_buckets (LSH bands)
/// - More bands = stricter matching (fewer false positives)
/// - Typical range: 10-50
/// - Documents must match in at least ONE band to be considered duplicates
///
/// ### bucket_size (hashes per band)
/// - More hashes = stricter matching within each band
/// - Typical range: 5-20
/// - Documents must match ALL hashes within a band
///
/// ### Total signature size
/// - `num_buckets × bucket_size` = total hash values per document
/// - Example: 20 bands × 5 hashes = 100 total hashes
/// - More total hashes = slower but more accurate
///
/// ### ngram_size
/// - Size of text chunks for comparison
/// - Typical values: 3 (trigrams), 5 (5-grams)
/// - Larger values = less sensitive to small differences
///
/// ### tokenizer
/// - "cl100k_base" or "cl100k": GPT-3.5/4 tokenizer (recommended)
/// - "p50k": GPT-3 tokenizer
/// - "uniseg": Unicode word boundary segmentation
/// - Default: Character-level (byte-based)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinHashParams {
    /// Number of LSH bands (default: 10)
    pub num_buckets: usize,
    /// Hash values per band (default: 20)
    pub bucket_size: usize,
    /// N-gram size for shingling (default: 5)
    pub ngram_size: usize,
    /// Random seed for hash permutations (default: 42)
    pub permutation_seed: u64,
    /// Tokenization strategy (default: "cl100k_base")
    pub tokenizer: String,
}

/// Engineering parameters for distributed processing.
///
/// These settings affect memory usage and file organization but don't
/// change the deduplication results.
///
/// ## Guidelines
///
/// ### num_docs
/// - **OVERESTIMATE** in multi-node settings
/// - Determines byte size for encoding document IDs
/// - Too small = encoding overflow errors
/// - Too large = slightly more disk space used
///
/// ### max_lines_per_path
/// - **OVERESTIMATE** in multi-node settings
/// - Determines byte size for encoding line numbers
/// - Should be >= largest file's line count
///
/// ### num_sig_chunks
/// - Number of signature file partitions
/// - Rule of thumb: ~100 per TB of data
/// - More chunks = better parallelism in edge gathering
/// - Typical range: 100-1000
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngParams {
    /// Expected total documents (default: 1 billion)
    /// **IMPORTANT**: Overestimate in multi-node settings!
    pub num_docs: usize,
    /// Maximum lines per file (default: 1 billion)
    /// **IMPORTANT**: Overestimate in multi-node settings!
    pub max_lines_per_path: usize,
    /// Number of signature chunks (default: 256)
    /// Recommended: ~100 per TB of data
    pub num_sig_chunks: usize,
}

/// Output behavior parameters.
///
/// Controls what happens to input files and how duplicates are handled.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputParams {
    /// Add duplicate metadata to documents (default: false)
    pub annotate: bool,
    /// JSON key for annotation metadata (default: "minhash.fuzzy")
    /// Can be nested, e.g., "metadata.dedup"
    pub annotate_key: String,
    /// Delete input files after processing (default: false)
    /// **WARNING**: Use with caution!
    pub delete_while_cleaning: bool,
    /// Remove duplicate documents (default: true)
    /// If false, all documents are kept with annotations
    pub remove_duplicates: bool,
}

/*=====================================================
=              Override Structures for CLI            =
=====================================================*/

/// Container for all configuration overrides from CLI arguments.
#[derive(Debug, Default)]
pub struct ConfigOverrides {
    pub minhash_params: MinHashOverrides,
    pub eng_params: EngOverrides,
    pub output_params: OutputOverrides,
}

/// CLI overrides for MinHash parameters.
///
/// `None` values indicate no override (use file or default).
#[derive(Debug, Default)]
pub struct MinHashOverrides {
    pub num_buckets: Option<usize>,
    pub bucket_size: Option<usize>,
    pub ngram_size: Option<usize>,
    pub permutation_seed: Option<u64>,
    pub tokenizer: Option<String>,
}

/// CLI overrides for engineering parameters.
#[derive(Debug, Default)]
pub struct EngOverrides {
    pub num_docs: Option<usize>,
    pub max_lines_per_path: Option<usize>,
    pub num_sig_chunks: Option<usize>,
}

/// CLI overrides for output parameters.
#[derive(Debug, Default)]
pub struct OutputOverrides {
    pub annotate: Option<bool>,
    pub annotate_key: Option<String>,
    pub delete_while_cleaning: Option<bool>,
    pub remove_duplicates: Option<bool>,
}

/*=====================================================
=                Default Implementations              =
=====================================================*/

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

/*=====================================================
=            Override Application Logic               =
=====================================================*/

impl ConfigOverrides {
    /// Applies all overrides to a configuration.
    ///
    /// Only non-None override values modify the config.
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
        if let Some(val) = self.annotate {
            config.annotate = val;
        }
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

/*=====================================================
=              Main Configuration Loading             =
=====================================================*/

impl Config {
    /// Loads configuration with optional file and CLI overrides.
    ///
    /// Resolution order:
    /// 1. Start with defaults
    /// 2. Apply YAML file if provided
    /// 3. Apply CLI overrides
    ///
    /// # Arguments
    /// * `config_path` - Optional path to YAML configuration file
    /// * `overrides` - CLI argument overrides
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

    /// Loads configuration from a YAML file only (no overrides).
    ///
    /// # Example
    /// ```no_run
    /// let config = Config::load_from_file(PathBuf::from("config.yaml"))?;
    /// ```
    pub fn load_from_file(config_path: PathBuf) -> Result<Self, Error> {
        Self::load_with_overrides(Some(config_path), ConfigOverrides::default())
    }

    /// Creates configuration from CLI overrides only (no file).
    ///
    /// Starts with defaults and applies only the specified overrides.
    pub fn from_overrides(overrides: ConfigOverrides) -> Self {
        let mut config = Self::default();
        overrides.apply_to(&mut config);
        config
    }
}
