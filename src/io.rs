/*
Some helpful utilities for i/o operations. Should be s3/local agnostic everywhere!


1. Listing files in a directory 
2. Reading files from s3/local into a vector of bytes
3. Writing a vector of bytes to s3/local
4. Compressing data


A note on extensions:
We will ONLY ever be concerned about files that have extensions of:
- .jsonl --> uncompressed, each line is a json string with a field 'text'
- .jsonl.gz -> jsonl compressed with gz
- .jsonl.zstd | .jsonl.zst -> jsonl compressed with zstandard 

Compression schemes will always be inferred from extension
*/

use std::fs::create_dir_all;
use std::fs::File;
use anyhow::Error;
use anyhow::anyhow;
use std::path::PathBuf;
use glob::glob;
use flate2::read::MultiGzDecoder;
use zstd::stream::read::Decoder as ZstdDecoder;
use std::io::{BufReader, Cursor, Write, Read};
use flate2::write::GzEncoder;
use flate2::Compression;
use zstd::stream::write::Encoder as ZstdEncoder;

const VALID_EXTS: &[&str] = &[".jsonl", ".jsonl.gz", ".jsonl.zstd", ".jsonl.zst", ".json.gz"];

/*======================================================================
=                              Listing files                           =
======================================================================*/




pub(crate) fn expand_dirs(paths: Vec<PathBuf>, manual_ext: Option<&[&str]>) -> Result<Vec<PathBuf>, Error> {
    // For local directories -> does a glob over each directory to get all files with given extension
    let exts = if !manual_ext.is_none() {
    	manual_ext.unwrap()
    } else {
    	VALID_EXTS
    };
    let mut files: Vec<PathBuf> = Vec::new();
    for path in paths {
        if path.is_dir() {
            let path_str = path
                .to_str()
                .ok_or_else(|| anyhow!("invalid path '{}'", path.to_string_lossy()))?;
        	for ext in exts {
        		let pattern = format!("{}/**/*{}", path_str, ext);
        		for entry in glob(&pattern).expect("Failed to read glob pattern") {
        			if let Ok(path) = entry {
        				files.push(path)
        			}
        		}
        	}
        } else {
            files.push(path.clone());
        }
    }
    Ok(files)
}


pub fn has_json_extension(path: &PathBuf) -> bool {
    if let Some(ext) = path.extension() {
        if ext == "json" {
            return true;
        } else if let Some(ext_os_str) = path.extension() {
            if let Some(ext_str) = ext_os_str.to_str() {
                return ext_str.starts_with("json.");
            }
        }
    }
    false
}


/*====================================================================
=                           Reading files                            =
====================================================================*/


pub(crate) fn read_pathbuf_to_mem(input_file: &PathBuf) -> Result<BufReader<Cursor<Vec<u8>>>, Error> {
    // Generic method to read local or s3 file into memory
    let contents = read_local_file_into_memory(input_file).expect("Failed to read contents into memory");
    let reader = BufReader::new(contents);

    Ok(reader)
} 



fn read_local_file_into_memory(input_file: &PathBuf) ->Result<Cursor<Vec<u8>>, Error>{
    // Takes a local file (must be local!) and reads it into a Cursor of bytes
    let mut file = File::open(input_file).expect("Failed to open file");

    let mut contents = Vec::new();
    let ext = input_file.extension().unwrap().to_string_lossy().to_lowercase();
    if ext == "gz" {
        // Gzip case        
        let mut decoder = MultiGzDecoder::new(file);
        decoder.read_to_end(&mut contents).expect("Failed to read local gz file");
    } else if ext == "zstd" || ext == "zst" {
        // Zstd case
        let mut decoder = ZstdDecoder::new(file).unwrap();
        decoder.read_to_end(&mut contents).expect("Failed to read local zstd file");
    } else {
        file.read_to_end(&mut contents).expect("Failed to read local file");

        // No compression case 
    }
    Ok(Cursor::new(contents))
}



/*====================================================================
=                          Writing files                             =
====================================================================*/

pub(crate) fn write_mem_to_pathbuf(contents: &[u8], output_file: &PathBuf) -> Result<(), Error> {
	let compressed_data = compress_data(contents.to_vec(), output_file);
    if let Some(parent_dir) = output_file.parent() {
            if !parent_dir.exists() {
                create_dir_all(parent_dir).unwrap()
             }
        let mut file = File::create(output_file).expect(format!("Unable to create output file {:?}", output_file).as_str());
        file.write_all(&compressed_data).expect(format!("Unable to write to {:?}", output_file).as_str());

    }
    Ok(())
}



fn compress_data(data: Vec<u8>, filename: &PathBuf) -> Vec<u8> {
    // Given a filename with an extension, compresses a bytestream accordingly 
    // {zst, zstd} -> zstandard, {gz} -> gzip, anything else -> nothing
    let output_data = match filename.extension().unwrap().to_str() {
        Some("gz") => {
            let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
            encoder.write_all(&data).unwrap();            
            encoder.finish().unwrap()
        },
        Some("zstd") | Some("zst") => {
            let mut encoder = ZstdEncoder::new(Vec::new(), 0).unwrap();
            encoder.write_all(&data).unwrap();            
            encoder.finish().unwrap()
        },
        _ => {data}
    };
    output_data
}


