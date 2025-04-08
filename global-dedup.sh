#!/bin/bash

# ===============================================================================
# Deduplication Script for Multiple Global Shards of RefinedWeb Data
# 
# This script implements a MinHash-based deduplication pipeline for large-scale
# text datasets. It processes data in multiple shards, generates signatures,
# identifies duplicates, and outputs deduplicated data.
# ===============================================================================

### build-file-map (lightweight)
# Creates file mappings for the input data shards
# Maps local and remote paths and creates separate filemaps for each global shard
build_filemaps() {
     # Get all file paths from S3
     s5cmd ls s3://ai2-llm/pretraining-data/sources/dclm/refinedweb/dolma_reformat/documents/* | grep -o "global-shard.*.jsonl.zstd$" | sort > files.txt
     # or (locally): find global-shard_*_of_10  -type f -name *.jsonl.zstd | sort > files.txt
     
     # Create filemap in correct json format
     jq -Rsc 'split("\n") | {local_input: "/mnt/raid0/input_data", remote_input: "UNNEEDED FOR SINGLE NODE", indices: (to_entries | map({value: .key, key: .value}) | from_entries)}' files.txt > filemap.json
     
     # Create filemaps for each global shard
     for shard_id in {01..10}; do 
          shard="global-shard_${shard_id}_of_10"
               jq '.indices |=  (to_entries | map(select(.key | startswith("'$shard'"))) | from_entries)' filemap.json -c > $shard.filemap.json
     done
     
     gzip -k *filemap.json

     for f in *filemap.json.gz; do
	     s5cmd cp -sp $f s3://ai2-llm/pretraining-data/sources/dclm/refinedweb/dolma_reformat/global_minhash_dedup/filemaps/;
     done
}


### hash-only (iterate through global shard)
# Processes each global shard to generate MinHash signatures
# Downloads data from S3, runs hash-only processing, and cleans up afterward
hash_only() {
     echo "$(date +"%Y-%m-%d %H:%M:%S")" "[start] hash_only"
     for shard_id in {01..10}; do 
     	shard="global-shard_${shard_id}_of_10"
     	echo "$(date +"%Y-%m-%d %H:%M:%S")" "Working on shard ${shard}"
     	mkdir -p /mnt/raid0/input_data/${shard}
     	mkdir -p /mnt/raid0/working_dir
     
     	s5cmd cp -sp s3://ai2-llm/pretraining-data/sources/dclm/refinedweb/dolma_reformat/global_minhash_dedup/filemaps/${shard}.filemap.json.gz  /mnt/raid0/working_dir/filemap.json.gz
     	s5cmd cp -sp s3://ai2-llm/pretraining-data/sources/dclm/refinedweb/dolma_reformat/documents/${shard}/*  /mnt/raid0/input_data/${shard}/
     
        cargo run --release -- hash-only --config examples/fineweb_global_config.yaml
     
     	mv /mnt/raid0/working_dir /mnt/raid0/working_dir_${shard_id}
     	
     	rm -r /mnt/raid0/input_data/${shard}
     done
     echo "$(date +"%Y-%m-%d %H:%M:%S")" "[done] has_only"
}

### gather-edges (globally)
# Merges working spaces from all shards and builds the duplicate detection graph
# Combines signatures from all shards, gathers edges between duplicates, and builds union-find structure
gather_edges() {
     echo "$(date +"%Y-%m-%d %H:%M:%S")" "[start] merge working spaces"
     mkdir -p /mnt/raid0/working_dir/sig_storage
     s5cmd cp -sp s3://ai2-llm/pretraining-data/sources/dclm/refinedweb/dolma_reformat/global_minhash_dedup/filemaps/filemap.json.gz  /mnt/raid0/working_dir/filemap.json.gz
      
     for sig_path in /mnt/raid0/working_dir_01/sig_storage/band_*/sigchunk_*; do 
     	sig_band=$(basename $(dirname $sig_path))
     	sig_chunk=$(basename $sig_path)
     	sig_chunk_path="/mnt/raid0/working_dir/sig_storage/$sig_band/$sig_chunk";
     	mkdir -p $sig_chunk_path
     
     	for shard_id in {01..10}; do
     	       	for sig_file in "/mnt/raid0/working_dir_${shard_id}/sig_storage/$sig_band/$sig_chunk/*.sig.bin"; do
     			ln -s $sig_file $sig_chunk_path/shard_${shard_id}_$(basename $sig_file);
     		done
     	done
     done
     echo "$(date +"%Y-%m-%d %H:%M:%S")" "[done] merge working spaces"
     
     echo "$(date +"%Y-%m-%d %H:%M:%S")" "[start] gather_edges"
     cargo run --release -- gather-edges --config examples/fineweb_global_config.yaml
     echo "$(date +"%Y-%m-%d %H:%M:%S")" "[done] gather_edges"
     
     echo "$(date +"%Y-%m-%d %H:%M:%S")" "[start] build_uf"
     cargo run --release -- build-uf --config examples/fineweb_global_config.yaml
     echo "$(date +"%Y-%m-%d %H:%M:%S")" "[done] build_uf"
     
     mv /mnt/raid0/working_dir /mnt/raid0/working_dir_global
}


### uf_size_prune (actually remove duplicates)
# Applies size-based pruning using the union-find structure to deduplicate data
# Processes each shard individually, but uses the global deduplication information
uf_size_prune() {
     echo "$(date +"%Y-%m-%d %H:%M:%S")" "[start] uf_size_prune"

     for shard_id in {01..10}; do
        echo "$(date +"%Y-%m-%d %H:%M:%S")" "Working on shard ${shard}"
        shard="global-shard_${shard_id}_of_10"
        mkdir -p /mnt/raid0/input_data/${shard}
        rm -r /mnt/raid0/working_dir/* || mkdir -p /mnt/raid0/working_dir

        ln -s /mnt/raid0/working_dir_global/* /mnt/raid0/working_dir
        rm /mnt/raid0/working_dir/filemap.json.gz
        
        s5cmd cp -sp s3://ai2-llm/pretraining-data/sources/dclm/refinedweb/dolma_reformat/documents/${shard}/*  /mnt/raid0/input_data/${shard}/

        cargo run --release -- uf-size-prune --config examples/fineweb_global_config.yaml

        s5cmd cp -sp /mnt/raid0/output_data/${shard} s3://ai2-llm/pretraining-data/sources/dclm/refinedweb/dolma_reformat-dedup/documents/${shard}/*

        rm -r /mnt/raid0/input_data/${shard}
	    rm -r /mnt/raid0/output_data/${shard}
     done 
     echo "$(date +"%Y-%m-%d %H:%M:%S")" "[done] uf_size_prune"
}

# Main execution pipeline
build_filemaps
hash_only
gather_edges
uf_size_prune
