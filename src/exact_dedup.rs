/* 
Some much-faster methods for exact-deduplication in rust. 
Can either annotate with a cc_id or remove
*/


use std::fs;
use anyhow::anyhow;
use std::collections::HashMap;
use std::io::BufRead;
use serde_json::{Value, json};
use dashmap::{DashMap, DashSet};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use anyhow::{Error, Result};
use rayon::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};
use rand::Rng;
use blake3;

use std::time::Instant;
use mj_io::{read_pathbuf_to_mem, build_pbar, write_mem_to_pathbuf, get_output_filename, expand_dirs};

use crate::storage::{FileMap, GenWriter };

const EOT_u64: u64 = u64::MAX;
const EOT: [u8;8] = EOT_u64.to_le_bytes();

/*===================================================================
=                            CONFIG STUFF                           =
===================================================================*/

#[derive(Debug, Serialize, Deserialize)]
struct ExactDedupConfig {
    name: String,
    local_input: PathBuf,
    output_dir: PathBuf,
    working_dir: PathBuf,
    text_field: String,

    #[serde(default)]
    annotate_only: bool

}


fn default_max_das_cc_size() -> usize {
    usize::MAX
}


#[derive(Debug, Serialize, Deserialize)]
struct DupAwareSubsampleConfig {
    name: String,
    local_input: PathBuf,
    output_dir: PathBuf,
    working_dir: PathBuf,
    cc_field: String,
   	subsample_rate: f32,
    #[serde(default = "default_max_das_cc_size")]
    max_cc_size: usize,   	

  
}


fn read_ed_config(config_path: &PathBuf) -> Result<ExactDedupConfig, Error> {
    let contents = read_pathbuf_to_mem(config_path).unwrap();
    let config: ExactDedupConfig = serde_yaml::from_reader(contents).unwrap();
    Ok(config)
}

fn read_das_config(config_path: &PathBuf) -> Result<DupAwareSubsampleConfig, Error> {
    let contents = read_pathbuf_to_mem(config_path).unwrap();
    let config: DupAwareSubsampleConfig = serde_yaml::from_reader(contents).unwrap();
    Ok(config)
}




/*===================================================================
=                           EXACT DEDUP MINHASH                     =
===================================================================*/

pub fn exact_dedup(config: &PathBuf, input_dir_override: Option<PathBuf>, output_dir_override: Option<PathBuf>) -> Result<(), Error> {
	println!("Starting exact dedup");
	let start_main = Instant::now();

	// Load the config and initialize things
	let mut config_obj = read_ed_config(config).unwrap();
	if let Some(input_dir) = input_dir_override {
		config_obj.local_input = input_dir;
	}
	if let Some(output_dir) = output_dir_override {
		config_obj.output_dir = output_dir;
	}

	let file_map = FileMap::new(&config_obj.local_input, &config_obj.local_input).unwrap();
    let this_chunk = file_map.get_path_chunk(0, 1);    

    // Hash all documents and store in a dashmap by their sigs
    let start_hash = Instant::now();
    println!("Starting doc hashing...");
    let pbar = build_pbar(this_chunk.len(), "Hashes");
    let doc_hash : DashMap<u64, Vec<(usize, usize)>> = DashMap::new();
    let total_docs : AtomicUsize = AtomicUsize::new(0);
    this_chunk.par_iter().for_each(|(path, path_id)| {
    	let input_path = config_obj.local_input.clone().join(path);
    	let line_count = exact_dedup_hash(&input_path, *path_id, &config_obj.text_field, &doc_hash).unwrap();
    	total_docs.fetch_add(line_count, Ordering::SeqCst);
    	pbar.inc(1);
    });
    let total_docs = total_docs.into_inner();
    let total_ccs = doc_hash.len();
    println!("Hashed {:?} docs in {:?} secs", total_docs, start_hash.elapsed().as_secs());


   	// Save intermed thing and  by path id
	let start_intermed = Instant::now();
	println!("Doing intermediate steps");
   	save_ccs_by_size(&doc_hash, &config_obj.working_dir).unwrap();
   	let lines_by_pathid = reorg_doc_hash(doc_hash, &config_obj.annotate_only).unwrap();
	println!("Intermed steps in {:?} secs", start_intermed.elapsed().as_secs());


   	// And then either annotate or scrub 
   	let start_modify = Instant::now();
   	println!("Starting file scrub...");


   	let pbar = build_pbar(this_chunk.len(), "Paths");
   	this_chunk.par_iter().for_each(|(path, path_id)| {
   		let input_path = config_obj.local_input.clone().join(path);
   		let output_path = get_output_filename(&input_path, &config_obj.local_input, &config_obj.output_dir).unwrap();
   		let exact_dedup_lines: Vec<(usize, u64, usize, usize)> = lines_by_pathid.remove(&path_id).unwrap().1;
   		scrub_file(&input_path, &output_path, exact_dedup_lines, &config_obj.annotate_only).unwrap();   
   		pbar.inc(1);
   	});


   	println!("Modified files in {:?} secs", start_modify.elapsed().as_secs());


   	println!("Saw {:?} docs and {:?} ccs | Removal rate would be {:?}",
   			 total_docs, total_ccs, (total_docs - total_ccs) as f32 / total_docs as f32);
   	println!("Finished full exact dedup flow in {:?} secs", start_main.elapsed().as_secs());


    Ok(())
}


fn exact_dedup_hash(path: &PathBuf, path_id: usize, text_field: &String, doc_hash: &DashMap<u64, Vec<(usize, usize)>>) -> Result<usize, Error> {
	let data = read_pathbuf_to_mem(path).unwrap();
	let mut line_count = 0;
	for (line_num, line) in data.lines().enumerate() {
        let line = line.unwrap();
        let json_obj: Value = serde_json::from_str(&line).expect(&format!("Failed to parse {:?} {:?}", path.clone(), line_num));
        let line_text = json_obj.get(text_field).unwrap().as_str().unwrap().to_string();
	    let mut hasher = DefaultHasher::new();
	    line_text.hash(&mut hasher);
    	let hash_val = hasher.finish();
    	doc_hash.entry(hash_val).or_default().push((path_id, line_num));
    	line_count += 1;
	}

	Ok(line_count)
}


fn reorg_doc_hash(doc_hash:  DashMap<u64, Vec<(usize, usize)>>, annotate_only: &bool) -> Result< DashMap<usize, Vec<(usize, u64, usize, usize)>>, Error> {
	// Output : path_id -> [(line_num, cc_id, cc_size, cc_idx)]
	// note: cc_idx is which element of the cc it is [useful for deduping later on]

	let lines_by_pathid : DashMap<usize, Vec<(usize, u64, usize, usize)>> = DashMap::new();
	let pbar = build_pbar(doc_hash.len(), "Ccs");

	doc_hash.par_iter().for_each(|entry| {
		let cc_id = entry.key();
		let doc_ids = entry.value();
		let cc_size = doc_ids.len();
		let mut idx = 0;
		for (path_id, line_num) in doc_ids.iter().skip( if *annotate_only {0} else {1}) {
			lines_by_pathid.entry(*path_id).or_default().push((*line_num, *cc_id, cc_size, idx));
			idx += 1;
		}
		pbar.inc(1);
	});

	Ok(lines_by_pathid)

}


fn scrub_file(input_path: &PathBuf, output_path: &PathBuf, lines: Vec<(usize, u64, usize, usize)>, annotate_only: &bool) -> Result<(), Error> {
	let data = read_pathbuf_to_mem(input_path).unwrap();
	let line_lookup : HashMap<usize, (usize, u64, usize)> = lines.into_iter().map(|(line_num, cc_id, cc_size, cc_idx)| (line_num, (cc_size, cc_id, cc_idx))).collect();

	let mut output_bytes : Vec<u8> = Vec::new();

	for (line_num, line) in data.lines().enumerate() {
        let line = line.unwrap();

        if *annotate_only {
	        let mut json_obj: Value = serde_json::from_str(&line).expect(&format!("Failed to parse {:?} {:?}", input_path.clone(), line_num));
	        let (cc_size, cc_id, cc_idx) = line_lookup.get(&line_num).unwrap();
	        let metadata = json!({"cc_id": cc_id, "cc_size": cc_size, "cc_idx": cc_idx});

	        json_obj.as_object_mut().unwrap().insert("exact_dedup".to_string(), metadata);
	        output_bytes.extend(serde_json::to_vec(&json_obj).unwrap());
	        output_bytes.push(b'\n');
        } else {
        	if !line_lookup.contains_key(&line_num) {
        		output_bytes.extend(line.as_bytes());
        		output_bytes.push(b'\n');
        	}
        }
	}

	if output_bytes.len() > 0 {
		write_mem_to_pathbuf(&output_bytes, &output_path).unwrap();
	}

	Ok(())
}


fn save_ccs_by_size(doc_hash: &DashMap<u64, Vec<(usize, usize)>>, working_dir: &PathBuf) -> Result<(), Error> {
	/* Saves a file working_dir/cc_size.bin 
	Which has file structure of the serialized list of ints
	[CC_SIZE, CC_id1, CC_id2, CC_id3, ..., EOT]
	where CC_id1, 2, 3 all have size CC_SIZE, and EOT is u64::max
	*/

	let pbar = 	build_pbar(doc_hash.len(), "CCs (intermed save 1/2)");
	let cc_groups : DashMap<usize, Vec<u64>> = DashMap::new();
	doc_hash.par_iter().for_each(|entry| {
		let cc_id = entry.key();
		let len = entry.value().len();
		cc_groups.entry(len).or_default().push(*cc_id);
		pbar.inc(1);
	});


	let pbar = build_pbar(cc_groups.len(), "CC sizes (intermed save 2/2)");
    let bin_contents: Vec<u8> = cc_groups.into_iter()
    	.par_bridge()
    	.flat_map(|(key, value)| {
    		let mut contents: Vec<u8> = Vec::new();
    		contents.extend(key.to_le_bytes());
    		let flat_val: Vec<u8> = value.into_iter().flat_map(|v| (v as u64).to_le_bytes()).collect();
    		contents.extend(flat_val);
    		contents.extend(EOT.clone());
    		pbar.inc(1);

    		contents
    	})
    	.collect();


    let cc_size_path = working_dir.clone().join("cc_size.bin");
   	write_mem_to_pathbuf(&bin_contents.as_slice(), &cc_size_path).unwrap();
	Ok(())

}






/*===========================================================================
=                          DUPLICATE AWARE SUBSAMPLING                      =
===========================================================================*/


fn duplicate_aware_subsample(config: &PathBuf, file_map: &FileMap) -> Result<(), Error> {
	let start_main = Instant::now();
	println!("Staring duplicate aware subsample...");

	// Load things we need 
	let start_setup = Instant::now();
	println!("Reading metadata to setup...");
	let config_obj = read_das_config(config).unwrap();
	let cc_by_size_path = config_obj.working_dir.clone().join("cc_size.bin");
	let cc_by_size = load_ccs_by_size(&cc_by_size_path).unwrap();
    let this_chunk = file_map.get_path_chunk(0, 1);    
    println!("Finished setup in {:?} secs", start_setup.elapsed().as_secs());

    // Subsample appropriately
    let start_survivors = Instant::now();
    println!("Making survivors...");
    let survivors = make_surviving_ccs(cc_by_size, config_obj.subsample_rate, config_obj.max_cc_size);
    println!("Made survivors in {:?} secs", start_survivors.elapsed().as_secs());


    // And then iterate over paths 
    let start_subsample = Instant::now();
    println!("Starting subsample...");
    let total_docs = AtomicUsize::new(0);
    let surviving_docs = AtomicUsize::new(0);
    let pbar = build_pbar(this_chunk.len(), "Paths");
    this_chunk.into_par_iter().for_each(|(p, _)| {
    	let output_path = get_output_filename(&p, &config_obj.local_input, &config_obj.output_dir).unwrap();
    	let (this_survive, this_total) = duplicate_aware_subsample_file(&p, &output_path, &survivors, &config_obj.cc_field).unwrap();
    	surviving_docs.fetch_add(this_survive, Ordering::SeqCst);
    	total_docs.fetch_add(this_total, Ordering::SeqCst);
    	pbar.inc(1);
    });

    let total_docs = total_docs.into_inner();
    let surviving_docs = surviving_docs.into_inner();
    println!("Subsampled data in {:?} secs", start_subsample.elapsed().as_secs());
    
    println!("Finished full dupaware sample in {:?} secs", start_main.elapsed().as_secs());
    println!("Kept {:?} docs of {:?} | Removal rate was {:?}", 
    		 surviving_docs, total_docs, (total_docs - surviving_docs) as f32 / total_docs as f32);
	Ok(())
}

fn load_ccs_by_size(cc_by_size_path : &PathBuf) -> Result<DashMap<u64, Vec<u64>>, Error> {
	let data = read_pathbuf_to_mem(cc_by_size_path).unwrap().into_inner().into_inner();
	let cc_by_size : DashMap<u64, Vec<u64>> = DashMap::new();

	let pbar = build_pbar(data.len() / 8, "ccbysize_chunks");
	let mut cc_size: u64 = EOT_u64;
	let mut cur_ids : Vec<u64> = Vec::new();
	for c in data.chunks(8) {
		let val = u64::from_le_bytes(c[..8].try_into().unwrap());
		if cc_size == EOT_u64 {
			cc_size = val;
			continue;
		} else if val == EOT_u64 {
			cc_by_size.insert(cc_size, cur_ids);
			cc_size = EOT_u64;
			cur_ids = Vec::new();
		} else {
			cur_ids.push(val);
		}
		pbar.inc(1);
	}


	Ok(cc_by_size)
}


fn make_surviving_ccs(ccs_by_size: DashMap<u64, Vec<u64>>, subsample_rate: f32, max_cc_size: usize) -> DashSet<u64> {
	let survivors: DashSet<u64> = DashSet::new();
	let pbar = build_pbar(ccs_by_size.len(), "Survivors");
	ccs_by_size.par_iter().for_each(|entry| {
	    let mut rng = rand::thread_rng();

	    let cc_size = *entry.key();
	    if cc_size as usize <= max_cc_size {	    	    
			for cc in entry.value() {
				if rng.gen::<f32>() < subsample_rate {
					survivors.insert(*cc);
				}
			}
		}
		pbar.inc(1);
	});

	survivors
}

fn duplicate_aware_subsample_file(path: &PathBuf, output_path: &PathBuf, surviving_ids: &DashSet<u64>, cc_field: &String) -> Result<(usize, usize), Error> {
	let data = read_pathbuf_to_mem(path).unwrap();

	let mut output_bytes: Vec<u8> = Vec::new();
	let mut total_docs = 0;
	let mut surviving_docs = 0;
	for line in data.lines() {
        let line = line.unwrap();    
        total_docs += 1;
        let line_json: Value = serde_json::from_str(&line).unwrap();
        let cc_id = line_json[cc_field]["cc_id"].as_u64().unwrap();
        if surviving_ids.contains(&cc_id) {
        	output_bytes.extend(line.as_bytes());
        	output_bytes.push(b'\n');
        	surviving_docs += 1;
        }
	}

	if output_bytes.len() > 0 {
		write_mem_to_pathbuf(&output_bytes, &output_path).unwrap();
	}

	Ok((surviving_docs, total_docs))
}



/*=============================================================================
=                                   MEGASCALE DUPAWARE                        =
=============================================================================*/


// Just a one-off thing to get signatures for all_dressed exact hashes. 
// Creates files like hash_signatures/CC_dump/sig_XXX.sig.bin 

// Where each entry is a (u32: path_id, u32: line_id, u128: signature)
// and sig_XXX has XXX ranging from 0..1023, where each line is the signature % 1024

pub fn get_exact_hash_signatures(config: &PathBuf, sig_prefix: &String, num_sig_chunks: usize) -> Result<(), Error> {
	let config_obj = read_ed_config(config).unwrap();
	let working_dir = config_obj.working_dir;
	let local_input = config_obj.local_input;
	let text_field = config_obj.text_field;
	let file_map = FileMap::load(&working_dir.clone().join("filemap.json.gz")).unwrap();
	let extensions = Some(&[".jsonl.zst"][..]);

	let files = expand_dirs(vec![local_input.clone()],
						    extensions).unwrap();
	let start_main = Instant::now();

	let sig_dir = working_dir.clone().join("u128_signatures").join(sig_prefix);
	let output_writer = GenWriter::new(&sig_dir, num_sig_chunks, "sig");

	println!("Starting hashing...");
	let pbar = build_pbar(files.len(), "Paths");
	files.par_iter().for_each(|p| {


		let contents = read_pathbuf_to_mem(&p).unwrap();
		for (line_num, line) in contents.lines().enumerate() {
			let path_stem = p.strip_prefix(local_input.clone()).ok().map(|stripped| stripped.strip_prefix("/").unwrap_or(stripped)).unwrap();
			let path_id =  file_map.indices.get(path_stem).unwrap();

			let line = line.unwrap();
			let json_obj: Value = serde_json::from_str(&line).unwrap();
			let contents = json_obj.get(text_field.clone()).unwrap().as_str().unwrap();
			let path_id_bytes = (*path_id as u32).to_le_bytes();
			let line_num_bytes = (line_num as u32).to_le_bytes();
			let hash = blake3::hash(contents.as_bytes());
			let bytes = hash.as_bytes();
			let mut result = 0u128;
			for i in 0..16 {
				result = (result << 8) | bytes[i] as u128;
			}
			let bucket = (result % (num_sig_chunks as u128)) as usize;
			let hash_val_bytes = result.to_le_bytes();

			let mut contents : Vec<u8> = Vec::new();
			contents.extend(path_id_bytes);
			contents.extend(line_num_bytes);
			contents.extend(hash_val_bytes); 
			output_writer.write_line(0, contents, bucket).unwrap();
		}
		pbar.inc(1);

	});
	output_writer.finish().unwrap();

	println!("Finished hash in {:?} secs", start_main.elapsed().as_secs());
	Ok(())
}

pub fn collate_cc_sizes(input_dir: &PathBuf, input_id: usize, output_dir: &PathBuf) -> Result<(), Error> {
	// Takes a bunch of .sig.bin files and loads them into memory 
	// And then counts occurrence of each cc_id (hash)
	// And writes a file .ccsize.bin with contents [(cc_id: u128, cc_size: u32)]
	let start_main = Instant::now();
	const CHUNK_SIZE : usize = 4 + 4 + 16;
	let cc_counter: DashMap<u128, u32> = DashMap::new();
	let custom_ext = format!("{:08}.sig.bin", input_id);
	let binding = [custom_ext.as_str()];
	let extensions = Some(&binding[..]);
	let paths = expand_dirs(vec![input_dir.clone()], extensions).unwrap();
	println!("PTAHS {:?}", paths);
	let last_p = paths.last().unwrap().clone();
	println!("Processing CC Signatures");
	let start_proc = Instant::now();
	let total_docs = AtomicUsize::new(0);
	paths.into_par_iter().for_each(|p| {
		let contents = read_pathbuf_to_mem(&p).unwrap().into_inner().into_inner();
		let num_chunks = contents.len() / CHUNK_SIZE;
		let pbar = if p == last_p {
			Some(build_pbar(num_chunks, "Path1 chunks"))
		} else {
			None
		};
		for chunk_id in 0..num_chunks {
			let chunk = u128::from_le_bytes(contents[chunk_id * CHUNK_SIZE + 8..chunk_id * CHUNK_SIZE + 24].try_into().unwrap());
			cc_counter.entry(chunk).and_modify(|c| *c += 1).or_insert(1);
			total_docs.fetch_add(1, Ordering::Relaxed);
			if let Some(ref pbar) = pbar {
				pbar.inc(1);
			}
		}
	});
	println!("Proc cc sigs in {:?} secs", start_proc.elapsed().as_secs());
	println!("Saw {:?} total docs | {:?} ccs", total_docs.into_inner(), cc_counter.len());
	// And then convert cc_sizes back to list
	let pbar = build_pbar(cc_counter.len(), "CC -> bytes");
	let cc_sizes: Vec<u8> = cc_counter.into_par_iter().flat_map(|(k,v)| {
		let mut row_bytes: Vec<u8> = Vec::new();
		row_bytes.extend(k.to_le_bytes()); // u128 -> 16 bytes
		row_bytes.extend(v.to_le_bytes()); // u32 -> 4 bytes
		// 20 bytes total
		pbar.inc(1);
		row_bytes
	}).collect();
	let output_path = output_dir.clone().join(format!("cc_sizes.{:08}.ccsize.bin", input_id));
	write_mem_to_pathbuf(&cc_sizes, &output_path).unwrap();

	println!("Finished collation in {:?} secs", start_main.elapsed().as_secs());

	Ok(())
}

pub fn annotate_file_ed(config: &PathBuf) -> Result<(), Error> {
	// loads the ccsize files, and loops over files to annotate with cc sizes
	let config_obj = read_ed_config(config).unwrap();
	let working_dir = config_obj.working_dir;
	let local_input = config_obj.local_input;
	let output_dir = config_obj.output_dir;
	let text_field = config_obj.text_field;


	let start_main = Instant::now();

	println!("Loading CC Sizes");
	let cc_size_exts = Some(&["ccsize.bin"][..]);
	let cc_size_files = expand_dirs(vec![working_dir.clone().join("cc_sizes")], cc_size_exts).unwrap();
	let cc_size_map = _load_ccsizes(&cc_size_files).unwrap();
	println!("Loaded {:?} cc sizes in {:?} secs", cc_size_map.len(), start_main.elapsed().as_secs());

	println!("Starting annotation...");
	let paths = expand_dirs(vec![local_input.clone()], None).unwrap();
	let pbar = build_pbar(paths.len(), "Input paths");
	paths.into_par_iter().for_each(|p| {
		let output_path = get_output_filename(&p, &local_input, &output_dir.clone()).unwrap();
		let data = read_pathbuf_to_mem(&p).unwrap();
		let mut output_bytes: Vec<u8> = Vec::new();
		for line in data.lines() {
			let line = line.unwrap();
			let mut json_obj: Value = serde_json::from_str(&line).unwrap();
			let contents = json_obj.get(text_field.clone()).unwrap().as_str().unwrap();
			let hash = blake3::hash(contents.as_bytes());
			let bytes = hash.as_bytes();
			let mut result = 0u128;
			for i in 0..16 {
				result = (result << 8) | bytes[i] as u128;
			}
			let cc_size = cc_size_map.get(&result).unwrap();
			let cc_json = json!({"size": *cc_size, "id": result});
			json_set(&mut json_obj, &"metadata.exact_dedup".to_string(), cc_json).unwrap();
			output_bytes.extend(serde_json::to_vec(&json_obj).unwrap());
			output_bytes.push(b'\n');
		}
		write_mem_to_pathbuf(&output_bytes, &output_path).unwrap();
		pbar.inc(1);
	});

	Ok(())
}


fn _load_ccsizes(cc_size_files: &Vec<PathBuf>) -> Result<DashMap<u128, u32>, Error> {
	let cc_size_map : DashMap<u128, u32> = DashMap::new();
	let pbar = build_pbar(cc_size_files.len(), "CC Size files");
	const CHUNK_SIZE: usize = 20;
	cc_size_files.par_iter().for_each(|p| {
		let contents = read_pathbuf_to_mem(p).unwrap().into_inner().into_inner();
		let num_chunks = contents.len() / CHUNK_SIZE;
		(0..num_chunks).into_iter().for_each(|i| {
			let chunk = &contents[i* CHUNK_SIZE.. i*CHUNK_SIZE + CHUNK_SIZE];
			let cc_id = u128::from_le_bytes(chunk[4..].try_into().unwrap());
			let cc_size = u32::from_le_bytes(chunk[..4].try_into().unwrap());
			cc_size_map.insert(cc_id, cc_size);
		});

		pbar.inc(1);
	});
	Ok(cc_size_map)
}




pub fn json_set(input: &mut Value, key: &String, val: Value) -> Result<(), Error> {
	let parts: Vec<&str> = key.split('.').collect();
	let mut current = input;

	for (i, &part) in parts.iter().enumerate() {
		if i == parts.len() - 1 {
			if current.is_object() {
				current[part] = val;
				return Ok(());
			} else {
				return Err(anyhow!("Weird nesting for setting json values"));
			}
		}
		if !current.is_object() {
			return Err(anyhow!("Weird nesting for setting json values"));
		}
		if !current.get(part).is_some() {
			current[part] = json!({});
		}
		current = &mut current[part];
	}
	Ok(())
}	



pub fn collect_dup_profile(cc_size_dir: &PathBuf, output: &PathBuf) -> Result<(), Error> {
	println!("Collecting duplicate profile.");
	let start_main = Instant::now();
	let cc_exts = Some(&["ccsize.bin"][..]);
	let cc_size_files = expand_dirs(vec![cc_size_dir.clone()], cc_exts).unwrap();
	const CHUNK_SIZE : usize = 20;

	let dup_profile: DashMap<u32, u64> = DashMap::new();
	let pbar = build_pbar(cc_size_files.len(), "CC Size files");
	cc_size_files.into_par_iter().for_each(|p| {
		let contents = read_pathbuf_to_mem(&p).unwrap().into_inner().into_inner();
		let num_chunks = contents.len() / CHUNK_SIZE;
		(0..num_chunks).into_iter().for_each(|i| {
			let chunk = &contents[i* CHUNK_SIZE.. i*CHUNK_SIZE + CHUNK_SIZE];
			//let cc_id = u128::from_le_bytes(chunk[4..].try_into().unwrap());
			let cc_size = u32::from_le_bytes(chunk[16..].try_into().unwrap());
			dup_profile.entry(cc_size).and_modify(|c| *c += 1).or_insert(1);
		});
		pbar.inc(1);
	});

	let pbar = build_pbar(dup_profile.len(), "Dup profile els");
	let dup_profile_bytes: Vec<u8> = dup_profile.into_par_iter().flat_map(|(k,v)| {
		let mut row_bytes: Vec<u8> = Vec::new();
		let cc_size = k.to_le_bytes();
		let cc_freq = v.to_le_bytes();
		row_bytes.extend(cc_size);
		row_bytes.extend(cc_freq);
		pbar.inc(1);
		row_bytes		
	}).collect();

	write_mem_to_pathbuf(&dup_profile_bytes, output).unwrap();
	println!("Made duplicate profile in {:?} secs", start_main.elapsed().as_secs());
	Ok(())
}


pub fn make_dupaware_sampler(cc_size_dir: &PathBuf, subsample_dir: &PathBuf, subsample_rate: f32, hard_max_size: usize, soft_max_size: usize) -> Result<(), Error> {
	println!("Making dupaware sampler...");
	let start_main = Instant::now();
	let cc_exts = Some(&["ccsize.bin"][..]);
	let cc_size_files = expand_dirs(vec![cc_size_dir.clone()], cc_exts).unwrap();
	const CHUNK_SIZE : usize = 20;


	let total_size = cc_size_files.iter().map(|p| fs::metadata(p).unwrap().len()).sum::<u64>() as usize;
	let total_chunks = total_size / CHUNK_SIZE;
	let pbar = build_pbar(total_chunks, "CCs");

	cc_size_files.into_par_iter().for_each(|p| {
		let contents = read_pathbuf_to_mem(&p).unwrap().into_inner().into_inner();
		let output_file = get_output_filename(&p, cc_size_dir, subsample_dir).unwrap();
		let output_file_str = output_file.to_string_lossy();
		let base = &output_file_str[.."ccsize.bin".to_string().len()];
		let output_file = PathBuf::from(format!("{}{}", base, "dupaware.bin"));

		let num_chunks = contents.len() / CHUNK_SIZE;
		let mut path_out: Vec<u8> = Vec::new();
		let mut rng = rand::thread_rng();
		(0..num_chunks).into_iter().for_each(|i| {
			let chunk = &contents[i* CHUNK_SIZE.. i*CHUNK_SIZE + CHUNK_SIZE];
			let cc_id = &chunk[4..];
			let cc_size = u32::from_le_bytes(chunk[..4].try_into().unwrap()) as usize;

			// Get the current subsample rate
			let cur_sub: f32 = if cc_size > hard_max_size {
				0.0
			} else if cc_size > soft_max_size {
				if rng.gen::<f32>() < subsample_rate {
					cc_size as f32 / soft_max_size as f32
				} else {
					0.0
				}
			} else {
				if rng.gen::<f32>() < subsample_rate {
					1.0
				} else {
					0.0
				}
			};
			path_out.extend(cc_id);
			path_out.extend(cur_sub.to_le_bytes());
			pbar.inc(1);

		});
		write_mem_to_pathbuf(&path_out, &output_file).unwrap();
	});

	println!("Made dupaware sampler in {:?} secs", start_main.elapsed().as_secs());
	

	Ok(())
}




