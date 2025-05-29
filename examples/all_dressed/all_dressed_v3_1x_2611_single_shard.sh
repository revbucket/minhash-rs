cargo run --release -- min-hash --config examples/all_dressed/all_dressed_v3_1x_2611.yaml > /mnt/raid0/minhash.log
cd ~/work/datamap-rs
cargo run --release -- reshard --input-dir /mnt/raid0/output --output-dir /mnt/raid0/resharded --max-size 256000000 > /mnt/raid0/reshard.log


s5cmd cp -sp /mnt/raid0/resharded/ s3://ai2-llm/pretraining-data/sources/cc_all_dressed/all_dressed_v3_subsamples/ed_sub0.25_minhash2611/
s5cmd cp -sp /mnt/raid0/*.log s3://ai2-llm/pretraining-data/sources/cc_all_dressed/all_dressed_v3_subsamples/ed_sub0.25_minhash2611/logs/
TOKEN=`curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600"`; INSTANCE_ID=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" -s http://169.254.169.254/latest/meta-data/instance-id); aws ec2 terminate-instances --instance-ids $INSTANCE_ID
