#!/bin/bash



# Store the input argument
X="02"


echo "Processing directory: $X"




# Step 0: preclean
echo "Pre-clean local storage..."
rm -rf "/mnt/raid0/input"
rm -rf "/mnt/raid0/output"
rm -rf "/mnt/raid0/working"
mkdir -p /mnt/raid0/logs_v2



# Step 1: Copy from S3 to local storage
echo "Copying data from S3 to local storage..."
s5cmd cp -sp "s3://ai2-llm/pretraining-data/sources/cc_all_dressed/all_dressed_v2/minhash_10shard_v3/download_scripts/part_${X}.txt" "/mnt/raid0/part_${X}.txt"
s5cmd run "/mnt/raid0/part_${X}.txt"

# Step 2: Run the map operation
echo "Running map operation..."
cd
# git clone https://github.com/revbucket/minhash-rs.git
cd ~/minhash-rs
git checkout lowermem_cc
git pull
cargo run --release -- min-hash --config examples/all_dressed/all_dressed_v2_10x.yaml > "/mnt/raid0/logs_v2/part_${X}_mh_output.log"
rm -rf /mnt/raid0/input
s5cmd cp -sp /mnt/raid0/output/ "s3://ai2-llm/pretraining-data/sources/cc_all_dressed/all_dressed_v2/minhash_10shard_v3/minhashed/part_${X}/"
s5cmd cp -sp "/mnt/raid0/logs_v2/" "s3://ai2-llm/pretraining-data/sources/cc_all_dressed/all_dressed_v2/minhash_10shard_v3/logs_v2/part_${X}/"


# Step 6: Clean up local storage
echo "Cleaning up local storage..."
rm -rf "/mnt/raid0/input"
rm -rf "/mnt/raid0/output"
rm -rf "/mnt/raid0/working"

echo "Processing complete for $X"