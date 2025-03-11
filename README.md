# Project Name

Superfast MinHash algorithms for fuzzy deduplication of .jsonl datasets. Natively implemented in Rust, with multinode orchestration handled via Python/Ray.

## Table of Contents
- [General Usage](#general-usage)
- [Install/Setup](#installsetup)
- [Single Node Usage](#single-node-usage)
- [Multinode Usage](#multinode-usage)
- [Release Notes](#release-notes)

## General Usage

I follow several general principles here:
- Rust code never interacts with remote file storage (e.g. S3)
- We assume we have access to a **lot** of really fast disk space, but reasonably small memory.
- The .jsonl* files in the dataset can be loaded into memory. Ideally they are small enough such that `num_cpus` files can live in memory at one time.
- Parameters for a particular setup are stored in a config file

With these assumptions in hand, the general workflow of how this tool works in a single-node setting is:
1. The set of files is collected and some renaming happens
2. Signatures are computed for every line of every file
3. Linked pairs of files are created and these "edges" are stored on the LFS
4. These "edges" are joined in a Union-Find structure to collect actual connected components, i.e., clusters of "fuzzy duplicates"
5. Fuzzy duplicates are either marked as such or pruned from the dataset.



## Install/Setup

The intent is to use this tool on EC2 instances, but any machine with many CPUs and plenty of local storage is helpful. I really like the EC2 i4i instances. 
A standard flow is to initialize an i4i.32xl instance and copy/paste the following code block:
```bash
sudo yum install mdadm -y
sudo mdadm --create /dev/md0 --level=0 --raid-devices=8 /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1 /dev/nvme4n1 /dev/nvme5n1 /dev/nvme6n1 /dev/nvme7n1 /dev/nvme8n1 
sudo mkfs.xfs /dev/md0
sudo mkdir /mnt/raid0
sudo mount /dev/md0 /mnt/raid0
sudo chown -R $USER /mnt/raid0

sudo yum install gcc -y
sudo yum install cmake -y
sudo yum install openssl-devel -y
sudo yum install g++ -y
sudo yum install htop -y
aws configure set aws_access_key_id <REDACTED>
aws configure set aws_secret_access_key <REDACTED>
aws configure set default.region <REDACTED>


wget https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz 
tar -xvzf s5cmd_2.2.2_Linux-64bit.tar.gz 
sudo mv s5cmd /usr/local/bin

sudo yum install git -y 
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > rustup.sh
bash rustup.sh -y
source ~/.bashrc
git clone https://github.com/revbucket/minhash-rs.git
cd minhash-rs
git checkout refac2025
cargo build --release 
```

This glues together 8 AWS Nitro drives making one really nice 27TB drive with very fast i/o. 
The next step is to download the data, for which I really like s5cmd (which should already be in your /bin/ if you copied the above code block).

Just do something like 
```
s5cmd cp -sp s3://bucket/path/to/jsonl/data/* /mnt/raid0/input_data
```



## Single Node Usage

If running on a single node, then you only need to have a `config.yaml` file and can just run with:
```
cargo run --release -- min-hash --config <path/to/config>
```

And that's it! End-to-end minhashing out of the box! 

If you want to do something more granular, you can do this step by step. Executed these commands one-by-one:
```
cargo run --release -- build-file-map --config <path/to/config>
cargo run --release -- hash-only --config <path/to/config>
cargo run --release -- gather-edges --config <path/to/config>
cargo run --release -- build-uf --config <path/to/config>
cargo run --release -- uf-size-prune --config <path/to/config> #NOTE: use command 'annotate' if you want to annotate instead
```



## Multinode Usage

Multinode usage is more complicated and should still be stress-tested. Let's call this a WIP for now.


## Release Notes

### v0.1.0 (YYYY-MM-DD)
- Initial release/write-up of README.md

