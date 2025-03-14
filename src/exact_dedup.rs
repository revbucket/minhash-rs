/* 
Some much-faster methods for exact-deduplication in rust. 
Can either annotate with a cc_id or remove
*/


use std::collections::HashMap;
use std::io::BufRead;
use serde_json::{Value, json};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use anyhow::{Error, Result};
use rayon::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};

use std::time::Instant;
use mj_io::{read_pathbuf_to_mem, build_pbar, write_mem_to_pathbuf};

use crate::storage::{FileMap};

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


fn read_config(config_path: &PathBuf) -> Result<ExactDedupConfig, Error> {
    let contents = read_pathbuf_to_mem(config_path).unwrap();
    let config: ExactDedupConfig = serde_yaml::from_reader(contents).unwrap();
    Ok(config)
}

fn get_output_filename(input_path: &PathBuf, config_input_dir: &PathBuf, config_output_dir: &PathBuf) -> Result<PathBuf, Error> {
	// Cloned fxn, maybe this goes to mj_io?
    let replaced = input_path.clone()
        .strip_prefix(config_input_dir)
        .ok()
        .map(|stripped| config_output_dir.clone().join(stripped)).unwrap();
    Ok(replaced)
}




/*===================================================================
=                           EXACT DEDUP MINHASH                     =
===================================================================*/

pub fn exact_dedup(config: &PathBuf, file_map: &FileMap) -> Result<(), Error> {
	println!("Starting exact dedup");
	let start_main = Instant::now();

	// Load the config and initialize things
	let config_obj = read_config(config).unwrap();
    let this_chunk = file_map.get_path_chunk(0, 1);    

    // Hash all documents and store in a dashmap by their sigs
    let start_hash = Instant::now();
    println!("Starting doc hashing...");
    let pbar = build_pbar(this_chunk.len(), "Hashes");
    let doc_hash : DashMap<u64, Vec<(usize, usize)>> = DashMap::new();
    let total_docs : AtomicUsize = AtomicUsize::new(0);
    this_chunk.par_iter().for_each(|(path, path_id)| {
    	let line_count = exact_dedup_hash(path, *path_id, &config_obj.text_field, &doc_hash).unwrap();
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
   		let output_path = get_output_filename(&path, &config_obj.local_input, &config_obj.output_dir).unwrap();
   		let exact_dedup_lines: Vec<(usize, u64, usize, usize)> = lines_by_pathid.remove(&path_id).unwrap().1;
   		scrub_file(&path, &output_path, exact_dedup_lines, &config_obj.annotate_only).unwrap();   
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
	let pbar = build_pbar(lines_by_pathid.len(), "Ccs");

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
	let eot = u64::MAX.to_le_bytes().to_vec();
    let bin_contents: Vec<u8> = cc_groups.into_iter()
    	.par_bridge()
    	.flat_map(|(key, value)| {
    		let mut contents: Vec<u8> = Vec::new();
    		contents.extend(key.to_le_bytes());
    		let flat_val: Vec<u8> = value.into_iter().flat_map(|v| (v as u64).to_le_bytes()).collect();
    		contents.extend(flat_val);
    		contents.extend(eot.clone());
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


