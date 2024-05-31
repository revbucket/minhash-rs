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



use std::fs::File;
use crate::s3::{get_reader_from_s3, expand_s3_dir, write_cursor_to_s3};
use anyhow::Error;
use anyhow::anyhow;
use std::path::PathBuf;
use crate::s3::is_s3;
use glob::glob;
use flate2::read::MultiGzDecoder;
use zstd::stream::read::Decoder as ZstdDecoder;
use std::io::{BufReader, Cursor, Write, Read};



const VALID_EXTS: &[&str] = &[".jsonl", ".jsonl.gz", ".jsonl.zstd", ".jsonl.zst"];

/*======================================================================
=                              Listing files                           =
======================================================================*/




pub(crate) fn expand_dirs(paths: Vec<PathBuf>) -> Result<Vec<PathBuf>, Error> {
    // For local directories -> does a glob over each directory to get all files with given extension
    // For s3 directories -> does an aws s3 ls to search for files
    let mut files: Vec<PathBuf> = Vec::new();
    let runtime = tokio::runtime::Runtime::new().unwrap();
    for path in paths {
        if is_s3(path.clone()) {
            // Use async_std to block until we scour the s3 directory for files
            runtime.block_on(async {
                let s3_paths = expand_s3_dir(&path, VALID_EXTS).await.unwrap();
                files.extend(s3_paths);                
            });                
        }
        else if path.is_dir() {
            let path_str = path
                .to_str()
                .ok_or_else(|| anyhow!("invalid path '{}'", path.to_string_lossy()))?;
        	for ext in VALID_EXTS {
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


/*====================================================================
=                           Reading files                            =
====================================================================*/


pub(crate) fn read_pathbuf_to_mem(input_file: &PathBuf) -> Result<BufReader<Cursor<Vec<u8>>>, Error> {
    // Generic method to read local or s3 file into memory
    let reader = if is_s3(input_file) {
        let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();   
        match rt.block_on(get_reader_from_s3(input_file, Some(5))) {
            Ok(result) => result,
            Err(err) => {
                eprintln!("Error! {:?}", err);
                return Err(err.into());
            }
        }
    } else {
        let contents = read_local_file_into_memory(input_file).expect("Failed to read contents into memory");
        BufReader::new(contents)
    };
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

pub(crate) fn get_output_filename(inputs: &[PathBuf], input_filename: &PathBuf, output_directory: &PathBuf) -> PathBuf {
    // More fancy output-file naming that no longer assumes unique inputs
    let matching_prefix = inputs
        .iter()
        .find(|pfx| input_filename.starts_with(pfx))
        .expect("No matching prefix found?!?");
    let relative_path = input_filename.strip_prefix(matching_prefix).unwrap();
    output_directory.clone().join(relative_path)
}


pub(crate) fn write_mem_to_pathbuf(contents: &[u8], output_file: &PathBuf) -> Result<(), Error> {
    if is_s3(output_file) {
        let cursor = Cursor::new(contents.to_vec());
        let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();   
        match rt.block_on(write_cursor_to_s3(&output_file, cursor)) {
            Ok(result) => result,
            Err(err) => {
                eprintln!("Error! {:?}", err);
                return Err(err.into());
            }
        };
    } else {
        let mut file = File::create(output_file).expect(format!("Unable to create output file {:?}", output_file).as_str());
        file.write_all(contents).expect(format!("Unable to write to {:?}", output_file).as_str());

    }
    Ok(())
}


