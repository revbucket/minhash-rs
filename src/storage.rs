/*
SmartER typing for storing minhash cliques.
For each band we want to store a dict:

signature -> [(path_id, line_num), ...]


With the specs:
signature: needs to be big enough to not allow collisions.
		   If E[collisions] is kept < 1, this means 1
		   Dynamically decided based on expected number of documents.
		   If # pairs is (#documents-choose-2), need number of bins to be greater than 
		   log_2(#documents-choose-2) 
		  
path_id: identifier for which path we are using
	defaults to int24 (unlikely we'll have >16MM paths, but highly likely we'll have >65k paths)

line_num: identifier for which line num within a file 
	defaults to int24 (unlikely we'll have >16MM paths, but highly likely we'll have >65k docs per path)

	where signature needs to be big enough to not allow collisions

*/
use std::hash::DefaultHasher;
use std::collections::HashMap;
use std::io::{BufWriter, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::fs::{OpenOptions, File, create_dir_all};
use std::path::{PathBuf};
use std::sync::{Arc, Mutex};
use std::cmp::{PartialEq, Eq};
use std::hash::{Hash, Hasher};
use dashmap::DashMap;
use anyhow::{Result, Error};
use mj_io::{read_pathbuf_to_mem, write_mem_to_pathbuf, expand_dirs};
use serde::{Deserialize, Serialize};
use rayon::prelude::*;




/*=========================================================
=                         Helpful byte math               =
=========================================================*/


pub(crate) fn to_byte_size(n: usize) -> usize {
	// Calculates number of bytes needed to store n unique elements
	let byte_size =(f64::log2(n as f64 ) / 8.0).ceil() as usize;
	if byte_size == 0 {
		1
	} else {
		byte_size
	}
}

pub(crate) fn compute_sig_size(num_docs: usize) -> usize {
	// Computes how many bytes needed to store the signatures such that collisions don't occur
	// Overflow might be tricky here so need to do math
	// log(n * (n-1) /2) = log(n) + log(n-1) - 1
	let log_num_pairs = f64::log2(num_docs as f64) + f64::log2(num_docs as f64 - 1.0) - 1.0;
	let sig_size = (log_num_pairs / 8.0).ceil() as usize;
	if sig_size == 0 {
		1
	} else {
		sig_size
	}

}



/*=========================================================
=                        Nonstandard int types            =
=========================================================*/
pub(crate) trait IntValue: Hash + Eq {
    fn new(value: usize) -> Self where Self: Sized;
    fn from_bytes(value: Vec<u8>) -> Self where Self: Sized;
    fn as_bytes(&self) -> &[u8];
    fn as_usize(&self) -> usize;
}

#[derive(PartialEq, Eq)]
#[derive(Clone, Copy)]
#[derive(Debug)]
pub(crate) struct IntN<const N: usize> {
    bytes: [u8; N],
}

impl<const N: usize> IntValue for IntN<N> {
    fn new(value: usize) -> Self {
        let mut bytes = [0; N];
        for i in 0..N {
            bytes[i] = (value >> ((N - 1 - i) * 8)) as u8;
        }
        IntN { bytes }
    }

    fn from_bytes(value: Vec<u8>) -> IntN<N> {
        let mut bytes = [0; N];
        for i in 0..N {
            bytes[i] = value[i];
        }
        IntN { bytes }
    }

    fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }    

    fn as_usize(&self) -> usize {
	    let mut result: usize = 0;
	    for &byte in &self.bytes {
	        result = (result << 8) | byte as usize;
	    }
	    result
    }
}

impl<const N: usize> Hash for IntN<N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.bytes.hash(state);
    }
}


#[derive(Clone, Debug)]
pub(crate) enum IntValueEnum {
	 Int8(IntN<1>),
	 Int16(IntN<2>),
	 Int24(IntN<3>),
	 Int32(IntN<4>),
	 Int40(IntN<5>),
	 Int48(IntN<6>),
	 Int56(IntN<7>),
	 Int64(IntN<8>),
	 Int72(IntN<9>),
	 Int80(IntN<10>),
	 Int88(IntN<11>),
	 Int96(IntN<12>),
	 Int104(IntN<13>),
	 Int112(IntN<14>),
	 Int120(IntN<15>),
	 Int128(IntN<16>),
}

impl IntValueEnum {
    pub(crate) fn new(value: usize, num_bytes: usize) -> Self {
        match num_bytes {
			1 => IntValueEnum::Int8(IntN::<1>::new(value)),
			2 => IntValueEnum::Int16(IntN::<2>::new(value)),
			3 => IntValueEnum::Int24(IntN::<3>::new(value)),
			4 => IntValueEnum::Int32(IntN::<4>::new(value)),
			5 => IntValueEnum::Int40(IntN::<5>::new(value)),
			6 => IntValueEnum::Int48(IntN::<6>::new(value)),
			7 => IntValueEnum::Int56(IntN::<7>::new(value)),
			8 => IntValueEnum::Int64(IntN::<8>::new(value)),
			9 => IntValueEnum::Int72(IntN::<9>::new(value)),
			10 => IntValueEnum::Int80(IntN::<10>::new(value)),
			11 => IntValueEnum::Int88(IntN::<11>::new(value)),
			12 => IntValueEnum::Int96(IntN::<12>::new(value)),
			13 => IntValueEnum::Int104(IntN::<13>::new(value)),
			14 => IntValueEnum::Int112(IntN::<14>::new(value)),
			15 => IntValueEnum::Int120(IntN::<15>::new(value)),
			16 => IntValueEnum::Int128(IntN::<16>::new(value)),
            // Add more cases for other IntN types
            _ => panic!("Unsupported IntN type"),
        }
    }

    pub(crate) fn from_bytes(value: Vec<u8>, num_bytes: usize) -> Self {
        match num_bytes {
			1 => IntValueEnum::Int8(IntN::<1>::from_bytes(value)),
			2 => IntValueEnum::Int16(IntN::<2>::from_bytes(value)),
			3 => IntValueEnum::Int24(IntN::<3>::from_bytes(value)),
			4 => IntValueEnum::Int32(IntN::<4>::from_bytes(value)),
			5 => IntValueEnum::Int40(IntN::<5>::from_bytes(value)),
			6 => IntValueEnum::Int48(IntN::<6>::from_bytes(value)),
			7 => IntValueEnum::Int56(IntN::<7>::from_bytes(value)),
			8 => IntValueEnum::Int64(IntN::<8>::from_bytes(value)),
			9 => IntValueEnum::Int72(IntN::<9>::from_bytes(value)),
			10 => IntValueEnum::Int80(IntN::<10>::from_bytes(value)),
			11 => IntValueEnum::Int88(IntN::<11>::from_bytes(value)),
			12 => IntValueEnum::Int96(IntN::<12>::from_bytes(value)),
			13 => IntValueEnum::Int104(IntN::<13>::from_bytes(value)),
			14 => IntValueEnum::Int112(IntN::<14>::from_bytes(value)),
			15 => IntValueEnum::Int120(IntN::<15>::from_bytes(value)),
			16 => IntValueEnum::Int128(IntN::<16>::from_bytes(value)),
            // Add more cases for other IntN types
            _ => panic!("Unsupported IntN type"),
        }
    }

    pub(crate) fn as_bytes(&self) -> &[u8] {
        match self {
            IntValueEnum::Int8(value) => value.as_bytes(),
            IntValueEnum::Int16(value) => value.as_bytes(),
            IntValueEnum::Int24(value) => value.as_bytes(),
            IntValueEnum::Int32(value) => value.as_bytes(),
            IntValueEnum::Int40(value) => value.as_bytes(),
            IntValueEnum::Int48(value) => value.as_bytes(),
            IntValueEnum::Int56(value) => value.as_bytes(),
            IntValueEnum::Int64(value) => value.as_bytes(),
            IntValueEnum::Int72(value) => value.as_bytes(),
            IntValueEnum::Int80(value) => value.as_bytes(),
            IntValueEnum::Int88(value) => value.as_bytes(),
            IntValueEnum::Int96(value) => value.as_bytes(),
            IntValueEnum::Int104(value) => value.as_bytes(),
            IntValueEnum::Int112(value) => value.as_bytes(),
            IntValueEnum::Int120(value) => value.as_bytes(),
            IntValueEnum::Int128(value) => value.as_bytes(),
            // Add more cases for other IntN types
        }

    }


    pub(crate) fn as_usize(&self) -> usize {
        match self {
            IntValueEnum::Int8(value) => value.as_usize(),
            IntValueEnum::Int16(value) => value.as_usize(),
            IntValueEnum::Int24(value) => value.as_usize(),
            IntValueEnum::Int32(value) => value.as_usize(),
            IntValueEnum::Int40(value) => value.as_usize(),
            IntValueEnum::Int48(value) => value.as_usize(),
            IntValueEnum::Int56(value) => value.as_usize(),
            IntValueEnum::Int64(value) => value.as_usize(),
            IntValueEnum::Int72(value) => value.as_usize(),
            IntValueEnum::Int80(value) => value.as_usize(),
            IntValueEnum::Int88(value) => value.as_usize(),
            IntValueEnum::Int96(value) => value.as_usize(),
            IntValueEnum::Int104(value) => value.as_usize(),
            IntValueEnum::Int112(value) => value.as_usize(),
            IntValueEnum::Int120(value) => value.as_usize(),
            IntValueEnum::Int128(value) => value.as_usize(),
            // Add more cases for other IntN types
        }

    }


}

impl PartialEq for IntValueEnum {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
			(IntValueEnum::Int8(a), IntValueEnum::Int8(b)) => a == b,
			(IntValueEnum::Int16(a), IntValueEnum::Int16(b)) => a == b,
			(IntValueEnum::Int24(a), IntValueEnum::Int24(b)) => a == b,
			(IntValueEnum::Int32(a), IntValueEnum::Int32(b)) => a == b,
			(IntValueEnum::Int40(a), IntValueEnum::Int40(b)) => a == b,
			(IntValueEnum::Int48(a), IntValueEnum::Int48(b)) => a == b,
			(IntValueEnum::Int56(a), IntValueEnum::Int56(b)) => a == b,
			(IntValueEnum::Int64(a), IntValueEnum::Int64(b)) => a == b,
			(IntValueEnum::Int72(a), IntValueEnum::Int72(b)) => a == b,
			(IntValueEnum::Int80(a), IntValueEnum::Int80(b)) => a == b,
			(IntValueEnum::Int88(a), IntValueEnum::Int88(b)) => a == b,
			(IntValueEnum::Int96(a), IntValueEnum::Int96(b)) => a == b,
			(IntValueEnum::Int104(a), IntValueEnum::Int104(b)) => a == b,
			(IntValueEnum::Int112(a), IntValueEnum::Int112(b)) => a == b,
			(IntValueEnum::Int120(a), IntValueEnum::Int120(b)) => a == b,
			(IntValueEnum::Int128(a), IntValueEnum::Int128(b)) => a == b,
            // Add more cases for other IntN types
            _ => panic!("Unsupported IntN type"),
        }
    }
}

impl Eq for IntValueEnum {}

impl Hash for IntValueEnum {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            IntValueEnum::Int8(a) => a.hash(state),
            IntValueEnum::Int16(a) => a.hash(state),
            IntValueEnum::Int24(a) => a.hash(state),
            IntValueEnum::Int32(a) => a.hash(state),
            IntValueEnum::Int40(a) => a.hash(state),
            IntValueEnum::Int48(a) => a.hash(state),
            IntValueEnum::Int56(a) => a.hash(state),
            IntValueEnum::Int64(a) => a.hash(state),
            IntValueEnum::Int72(a) => a.hash(state),
            IntValueEnum::Int80(a) => a.hash(state),
            IntValueEnum::Int88(a) => a.hash(state),
            IntValueEnum::Int96(a) => a.hash(state),
            IntValueEnum::Int104(a) => a.hash(state),
            IntValueEnum::Int112(a) => a.hash(state),
            IntValueEnum::Int120(a) => a.hash(state),
            IntValueEnum::Int128(a) => a.hash(state),
            // Add more cases for other IntN types
        }
    }
}




/*=================================================================
=                               PATH LOOKUP                       =
=================================================================*/
/*
Struct that saves the order of files. Useful for making sure we have the same order.
Basically just used to map filenames to usizes and vice versa so we only have to
pass around indices vs whole strings
*/
pub struct PathLookup {
    //paths: Vec<PathBuf>,
    pub indices: DashMap<PathBuf, usize>,
}

impl PathLookup {
    pub fn new(paths: Vec<PathBuf>) -> Self {
        let indices = Self::build_indices(&paths);
        PathLookup {// paths, 
                    indices }
    }

    pub fn build_indices(paths: &Vec<PathBuf>) -> DashMap<PathBuf, usize> {
        paths
            .iter()
            .enumerate()
            .map(|(index, path)| (path.clone(), index))
            .collect()
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn get_chunk(&self, chunk_id: usize, num_chunks: usize) -> Vec<(PathBuf, usize)>{
        let chunk : Vec<(PathBuf, usize)> = self.indices.iter()
             .filter(|entry| entry.value() % num_chunks == chunk_id)
             .map(|entry| (entry.key().clone(), *entry.value()))
             .collect();
        chunk
    }

    pub fn save_to_file(&self, file_path: &PathBuf) -> Result<(), Error> {
        let hash_indices: HashMap<_, _> = self.indices.clone().into_par_iter().collect();
        let serialized = serde_json::to_vec(&hash_indices)?;
        write_mem_to_pathbuf(&serialized, file_path)
    }

    pub fn load_from_file(file_path: &PathBuf) -> Result<Self> {
        let contents = read_pathbuf_to_mem(file_path).unwrap();
        let deserialized: HashMap<PathBuf, usize> = serde_json::from_reader(contents)?;
        let mut pairs: Vec<(PathBuf, usize)> = deserialized.into_iter().collect();
        pairs.sort_by_key(|&(_, id)| id);
        let paths = pairs.into_iter().map(|(path, _)| path).collect();
        Ok(PathLookup::new(paths))
    }

}


/*==========================================================
=                     MinHash Config                       =
==========================================================*/
/*
Object that should be built for EVERY MinHash run.
Holds:
	- the specified directories for input 
	- the mapping of filenames -> indices 
	- the number of bytes required to store a 
		+ signature
		+ path 
		+ line number

And supports methods to:
- load/save from file/s3
- get "path chunks" 
- access attributes (i.e., number of paths)
*/

#[derive(Serialize, Deserialize)]
pub struct MinHashConfig {
	pub input: Vec<PathBuf>,
	pub indices: HashMap<PathBuf, usize>,
	pub sig_size: usize,
	pub path_size: usize,
	pub line_size: usize
}

impl MinHashConfig {
	pub fn new(input: &Vec<PathBuf>, num_docs: usize, max_lines_per_path: usize) -> Result<Self, Error> {
		// First gather all files 
		let paths = expand_dirs(input.clone(), None).unwrap();
		let num_paths = paths.len();
		let path_size = to_byte_size(num_paths);
		let line_size = to_byte_size(max_lines_per_path);
		let sig_size = compute_sig_size(num_docs);
		let indices : HashMap<PathBuf, usize> = paths.iter()
			.enumerate()
			.map(|(i, p)| (p.clone(), i))
			.collect();
		Ok(MinHashConfig {input: input.clone(),
						  indices, 
						  sig_size,
						  path_size,
						  line_size})
	}

	pub fn save(&self, save_loc: &PathBuf) -> Result<(), Error> {
		let json_bytes = serde_json::to_vec(self).unwrap();
		write_mem_to_pathbuf(&json_bytes, &save_loc)
	}

	pub fn load(load_loc: &PathBuf) -> Result<Self, Error> {
		let json_bytes = read_pathbuf_to_mem(&load_loc).unwrap();
		let cursor = json_bytes.into_inner();
		let binding = cursor.into_inner();
		let contents = binding.as_slice();
		let config: MinHashConfig = serde_json::from_slice(&contents).unwrap();		
		Ok(config)
	}

	pub fn get_chunk(&self, chunk_id: usize, num_chunks: usize) -> Vec<(PathBuf, usize)> {
        let chunk : Vec<(PathBuf, usize)> = self.indices.iter()
             .filter(|(_k, v)| *v % num_chunks == chunk_id)
             .map(|(k, v)| (k.clone(), *v))
             .collect();
        chunk		
	}

	pub fn len(&self) -> usize {
		self.indices.len()
	}
}



#[derive(Serialize, Deserialize)]
pub struct FileMap {
	pub local_input: PathBuf,
	pub remote_input: PathBuf,
	pub indices: HashMap<PathBuf, usize>	
}

impl FileMap {
	pub fn new(local_input: &PathBuf, remote_input: &PathBuf) -> Result<Self, Error> {
		let input_vec = vec![local_input.clone()];
		let paths = expand_dirs(input_vec.clone(), None).unwrap();

		let stripped_paths: Vec<PathBuf> = paths.iter().map(|p| p.strip_prefix(local_input.clone()).unwrap().to_path_buf()).collect();

		let indices : HashMap<PathBuf, usize> = stripped_paths.iter()
			.enumerate()
			.map(|(i, p)| (p.clone(), i))
			.collect();
		Ok(FileMap {local_input: local_input.clone(),
				    remote_input: remote_input.clone(),				    
					indices})
	}

	pub fn save(&self, save_loc: &PathBuf) -> Result<(), Error> {
		let json_bytes = serde_json::to_vec(self).unwrap();
		write_mem_to_pathbuf(&json_bytes, &save_loc)
	}

	pub fn load(load_loc: &PathBuf) -> Result<Self, Error> {
		let json_bytes = read_pathbuf_to_mem(&load_loc).unwrap();
		let cursor = json_bytes.into_inner();
		let binding = cursor.into_inner();
		let contents = binding.as_slice();
		let filemap: FileMap = serde_json::from_slice(&contents).unwrap();
		Ok(filemap)
	}


	pub fn get_path_chunk(&self, chunk_id: usize, num_chunks: usize) -> Vec<(PathBuf, usize)> {
		let chunk : Vec<(PathBuf, usize)> = self.indices.iter()
			.filter(|(_k, v)| *v % num_chunks == chunk_id)
			.map(|(k, v)| (k.clone(), *v))
			.collect();
		chunk
	}

	pub fn len(&self) -> usize {
		self.indices.len()
	}
}


/*==========================================================
=                     Band Storage Config.                 =
==========================================================*/
#[derive(Serialize, Deserialize)]
pub struct BandStorageConfig {
	pub sig_size: usize,
	pub path_size: usize,
	pub line_size: usize,
}

impl BandStorageConfig {
	pub fn new(sig_size: usize, path_size: usize, line_size: usize) -> Self {
		BandStorageConfig {sig_size, path_size, line_size}
	}

	pub fn infer_new(num_docs: usize, num_paths: usize, max_lines_per_doc: usize) -> Self {
		let sig_size = compute_sig_size(num_docs);
		let path_size = to_byte_size(num_paths);
		let line_size = to_byte_size(max_lines_per_doc);
		BandStorageConfig {sig_size, path_size, line_size}
	}

	pub fn save(&self, save_loc: PathBuf) -> Result<(), Error> {
		// Save to json here 
		let json_bytes = serde_json::to_vec(self).unwrap();
		write_mem_to_pathbuf(&json_bytes, &save_loc)
	}

	pub fn load(load_loc: &PathBuf) -> Result<Self, Error> {
		// Load from json at load_loc
		let json_bytes = read_pathbuf_to_mem(&load_loc).unwrap();
		let cursor = json_bytes.into_inner();
		let binding = cursor.into_inner();
		let contents = binding.as_slice();
		let config: BandStorageConfig = serde_json::from_slice(&contents).unwrap();		
		Ok(config)
	}
}




/*==========================================================
=                     Signature Writer                     =
==========================================================*/
pub struct SignatureWriter {
	pub writer: DashMap<(u32, usize), Arc<Mutex<BufWriter<File>>>>,
	storage_loc: PathBuf,
	band_ids: Vec<u32>,
	num_sig_chunks: usize,
}

impl SignatureWriter {
	pub fn new(storage_loc: &PathBuf, band_ids: Vec<u32>, num_sig_chunks: usize, path_chunk: usize) -> Self {
		let writer : DashMap<(u32, usize), Arc<Mutex<BufWriter<File>>>> = DashMap::new();
		// Create writers into |band_ids|
		println!("Opening {:?} signature files", band_ids.len() * num_sig_chunks);
		for band_id in &band_ids {
			for sig_chunk in 0..num_sig_chunks {
				let filename = SignatureWriter::get_filename(storage_loc, *band_id, sig_chunk, path_chunk);
				if let Some(parent_dir) = filename.parent() {
			        if !parent_dir.exists() {
			            create_dir_all(parent_dir).unwrap()
			         }
			    }
				let sigwriter = Arc::new(
					Mutex::new(
					BufWriter::new(
					OpenOptions::new()
					.append(true)
					.create(true)
					.mode(0o644)
					.open(filename)
					.unwrap()
				)));
				writer.insert((*band_id, sig_chunk), sigwriter);			
			}
		}
		SignatureWriter { writer, storage_loc: storage_loc.clone(), band_ids: band_ids.clone(), num_sig_chunks }
	}

	pub fn get_filename(storage_loc: &PathBuf, band_id: u32, sig_chunk: usize, path_chunk: usize) -> PathBuf {
		storage_loc.clone()
			.join(format!("band_{:016}", band_id))
			.join(format!("sigchunk_{:08}", sig_chunk))
			.join(format!("pathchunk_{:08}.sig.bin", path_chunk))
	}

	pub fn get_input_output_filenames(&self, output_loc: &PathBuf, path_chunk: usize) -> Vec<(PathBuf, PathBuf)> {
		let mut io_pairs : Vec<(PathBuf, PathBuf)> = Vec::new();
		for band_id in &self.band_ids {
			for sig_chunk in 0..self.num_sig_chunks {
				let input_filename = SignatureWriter::get_filename(&self.storage_loc, *band_id, sig_chunk, path_chunk);
				let output_filename = SignatureWriter::get_filename(output_loc, *band_id, sig_chunk, path_chunk);
				io_pairs.push((input_filename, output_filename));
			}
		}
		io_pairs
	}

	pub fn write_line(&self, band_id: u32, sig_chunk: usize, contents: Vec<u8>) -> Result<(), Error> {
		let key = (band_id, sig_chunk);
		let binding = self.writer.get(&key).unwrap();
		let mut sigwriter = binding.lock().unwrap();
		sigwriter.write_all(&contents).unwrap();

		Ok(())
	}

	pub fn finish(&self) -> Result<(), Error> {
		// Flushes all the open writers
		self.writer.par_iter()
			.for_each(|entry| entry.value().lock().unwrap().flush().unwrap());
		Ok(())
	}
}


/*======================================================
=                       CC Writer                      =
======================================================*/

pub struct GenWriter {
	pub writer: DashMap<usize, Arc<Mutex<BufWriter<File>>>>,
	#[allow(dead_code)]
	storage_loc: PathBuf,
	num_chunks: usize,
}

impl GenWriter {
	pub fn new(storage_loc: &PathBuf, num_chunks: usize, subext: &str) -> Self {
		let writer : DashMap<usize, Arc<Mutex<BufWriter<File>>>> = DashMap::new();
		// Create writers
		println!("Opening {:?} writer files", num_chunks);
		for chunk in 0..num_chunks {
			let filename = GenWriter::get_filename(storage_loc, chunk, subext);
			if let Some(parent_dir) = filename.parent() {
		        if !parent_dir.exists() {
		            create_dir_all(parent_dir).unwrap()
		         }
		    }
			let ccwriter = Arc::new(
				Mutex::new(
				BufWriter::new(
				OpenOptions::new()
				.append(true)
				.create(true)
				.mode(0o644)
				.open(filename)
				.unwrap()
			)));
			writer.insert(chunk, ccwriter);
		}
		GenWriter { writer, storage_loc: storage_loc.clone(), num_chunks: num_chunks}
	}


	pub fn get_filename(storage_loc: &PathBuf, chunk: usize, subext: &str) -> PathBuf {
		storage_loc.clone()
			.join(format!("chunk_{:08}.{}.bin", chunk, subext))
	}


	pub fn write_line(&self, key: usize, contents: Vec<u8>, force_key: usize) -> Result<(), Error> {
		// hash the key and take mod num_chunks to get location


	    let chunk_id = if force_key > 0 {
	    	force_key
	    } else {
		    let mut hasher = DefaultHasher::new();
		    key.hash(&mut hasher);	    	
	    	hasher.finish() as usize % self.num_chunks
	    };


		let binding = self.writer.get(&chunk_id).unwrap();
		let mut cc_writer = binding.lock().unwrap();
		cc_writer.write_all(&contents).unwrap();
		
		Ok(())

	}

	pub fn finish(&self) -> Result<(), Error> {
		// Flushes all the open writers
		self.writer.par_iter()
			.for_each(|entry| entry.value().lock().unwrap().flush().unwrap());
		Ok(())
	}
}



