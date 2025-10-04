#!/bin/bash



# Store the input argument
X=08


echo "Processing directory: $X"

mkdir -p /mnt/raid0/logs/


# Step 0: preclean
echo "Pre-clean local storage..."
rm -rf "/mnt/raid0/input"
rm -rf "/mnt/raid0/output"
rm -rf "/mnt/raid0/working"

echo "Copying data from S3 to local storage..."
s5cmd cp s3://ai2-llm/pretraining-data/sources/cc_all_dressed/all_dressed_v3/minhash/param_26_11/working/filemap.json.gz /mnt/raid0/working/
s5cmd cp -sp "s3://ai2-llm/pretraining-data/sources/cc_all_dressed/all_dressed_v3/exact_dedup_v2/${X}/*" /mnt/raid0/input/${X}/

cd ~/minhash-rs
git checkout lowermem_cc2
git pull
cargo run --release -- hash-only --config examples/all_dressed/all_dressed_v3_1x_2611.yaml --id-subext $X > "/mnt/raid0/logs/hashing_part_${X}.log"


s5cmd cp -sp /mnt/raid0/working/sig_storage/ s3://ai2-llm/pretraining-data/sources/cc_all_dressed/all_dressed_v3/minhash/param_26_11/working/sig_storage/
s5cmd cp -sp /mnt/raid0/logs/* s3://ai2-llm/pretraining-data/sources/cc_all_dressed/all_dressed_v3/minhash/param_26_11/

# Step 6: Clean up local storage
echo "Cleaning up local storage..."
rm -rf "/mnt/raid0/input"
rm -rf "/mnt/raid0/output"
rm -rf "/mnt/raid0/working"

echo "Processing complete for $X"	