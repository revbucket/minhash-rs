/* 
Some much-faster methods for exact-deduplication in rust. 
Can either annotate with a cc_id or remove
*/


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

use std::time::Instant;
use mj_io::{read_pathbuf_to_mem, build_pbar, write_mem_to_pathbuf, get_output_filename};

use crate::storage::{FileMap};

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

