use std::io::BufRead;
use serde_json::Value;
use dashmap::{DashMap, DashSet};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use anyhow::{Error, Result};
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use rand::Rng;

use std::time::Instant;
use mj_io::{read_pathbuf_to_mem, build_pbar, write_mem_to_pathbuf};

use crate::storage::{FileMap};

#[allow(non_upper_case_globals)]
const EOT_u64: u64 = u64::MAX;



/*============================================================================
=                                   UTILITIES                                =
============================================================================*/


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


fn read_das_config(config_path: &PathBuf) -> Result<DupAwareSubsampleConfig, Error> {
    let contents = read_pathbuf_to_mem(config_path).unwrap();
    let config: DupAwareSubsampleConfig = serde_yaml::from_reader(contents).unwrap();
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






/*===========================================================================
=                          DUPLICATE AWARE SUBSAMPLING                      =
===========================================================================*/


pub fn duplicate_aware_subsample(config: &PathBuf) -> Result<(), Error> {
	let start_main = Instant::now();
	println!("Staring duplicate aware subsample...");

	// Load things we need 
	let start_setup = Instant::now();
	println!("Reading metadata to setup...");
	let config_obj = read_das_config(config).unwrap();
	let file_map = FileMap::new(&config_obj.local_input, &config_obj.local_input).unwrap();
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

