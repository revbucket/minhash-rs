#!/bin/bash



# Store the input argument
X="XX"


echo "Processing directory: $X"




# Step 0: preclean
echo "Pre-clean local storage..."
rm -rf "/mnt/raid0/input"
rm -rf "/mnt/raid0/output"
rm -rf "/mnt/raid0/working"




# Step 1: Copy from S3 to local storage
echo "Copying data from S3 to local storage..."
sc5md cp -sp "s3://ai2-llm/pretraining-data/sources/cc_all_dressed/all_dressed_v2/minhash_10shard/download_scripts/part_${X}.txt" "/mnt/raid0/part_${X}.txt"
s5cmd run "/mnt/raid0/part_${X}.txt"

# Step 2: Run the map operation
echo "Running map operation..."
cd
# git clone https://github.com/revbucket/minhash-rs.git
cd ~/minhash-rs
git checkout lowermem_cc

cargo run --release -- min-hash --config examples/all_dressed/all_dressed_v2_10x.yaml > "/mnt/raid0/part_${X}_output.log"


# Step 5: Copy results back to S3
# S3 file structure looks like ... :
# s3://ai2-llm/pretraining-data/sources/cc_all_dressed/
#     - all_dressed_v2/english/{CC_DUMP}/*.jsonl.*
#     - all_dressed_v2/logs/{CC_DUMP}/*.txt

echo "Copying results back to S3..."
s5cmd cp -sp /mnt/raid0/output/ "s3://ai2-llm/pretraining-data/sources/cc_all_dressed/all_dressed_v2/minhash_10shard/output_${X}/"
s5cmd cp -sp "/mnt/raid0/part_${X}_output.log" "s3://ai2-llm/pretraining-data/sources/cc_all_dressed/all_dressed_v2/minhash_10shard/logs/part_${X}_output.log"

# Step 6: Clean up local storage
echo "Cleaning up local storage..."
rm -rf "/mnt/raid0/input"
rm -rf "/mnt/raid0/output"
rm -rf "/mnt/raid0/working"

echo "Processing complete for $X"av