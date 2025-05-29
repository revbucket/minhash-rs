cargo run --release -- min-hash --config examples/all_dressed/all_dressed_v3_1x_2611.yaml > /mnt/raid0/minhash.log
cd ~/work/datamap-rs
git checkout sort_v2
git pull
cargo run --release -- dist-group --input-dir /mnt/raid0/output --group-dir /mnt/raid0/groups --config examples/all_dressed/minhash_groupsort.yaml > /mnt/raid0/group.log
cargo run --release -- dist-sort --input-dir /mnt/raid0/groups --output-dir /mnt/raid0/sorted --config examples/all_dressed/minhash_groupsort.yaml > /mnt/raid0/sort.log
cargo run --release -- group-sort-filter --input-dir /mnt/raid0/sorted --output-dir /mnt/raid0/filter --config examples/all_dressed/minhash_groupsort.yaml > /mnt/raid0/filter.log

s5cmd cp -sp /mnt/raid0/groups/ s3://ai2-llm/pretraining-data/sources/cc_all_dressed/all_dressed_v3_subsamples/ed_sub0.25_minhash2611/groups/
s5cmd cp -sp /mnt/raid0/sorted/ s3://ai2-llm/pretraining-data/sources/cc_all_dressed/all_dressed_v3_subsamples/ed_sub0.25_minhash2611/sorted/
s5cmd cp -sp /mnt/raid0/filter/ s3://ai2-llm/pretraining-data/sources/cc_all_dressed/all_dressed_v3_subsamples/ed_sub0.25_minhash2611/filter/

s5cmd cp -sp /mnt/raid0/*.log s3://ai2-llm/pretraining-data/sources/cc_all_dressed/all_dressed_v3_subsamples/ed_sub0.25_minhash2611/logs/



sleep 60
TOKEN=`curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600"`; INSTANCE_ID=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" -s http://169.254.169.254/latest/meta-data/instance-id); aws ec2 terminate-instances --instance-ids $INSTANCE_ID
