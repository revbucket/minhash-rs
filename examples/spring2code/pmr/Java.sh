#!/bin/bash



# Store the input argument
X="Java"


echo "Processing directory: $X"




# Step 0: preclean
echo "Pre-clean local storage..."
rm -rf "/mnt/raid0/*"


mkdir -p /mnt/raid0/logs_v2



# Step 1: Copy from S3 to local storage
echo "Copying data from S3 to local storage..."
s5cmd cp "s3://ai2-llm/pretraining-data/sources/the-stack-v2/spring2code_v2/data/${X}/*" /mnt/raid0/input/

# Step 2: Run the map operation
echo "Running map operation..."
cd
# git clone https://github.com/revbucket/minhash-rs.git
cd ~/minhash-rs
git checkout lowermem_cc2
git pull
cargo run --release -- min-hash --config examples/spring2code/config.yaml > "/mnt/raid0/minhash_${X}.log"
rm -rf /mnt/raid0/input
s5cmd cp -sp /mnt/raid0/output/ "s3://ai2-llm/pretraining-data/sources/the-stack-v2/spring2code_v2/minhash_v2/data/${X}/"
s5cmd cp -sp "/mnt/raid0/*.log" "s3://ai2-llm/pretraining-data/sources/the-stack-v2/spring2code_v2/minhash_v2/logs/"


# Step 6: Clean up local storage
echo "Cleaning up local storage..."
rm -rf "/mnt/raid0/input"
rm -rf "/mnt/raid0/output"
rm -rf "/mnt/raid0/working"

echo "Processing complete for $X"