# Fuzzy Deduplication (Multi-Node)

A test case to run fuzzy deduplication in a "Multi-node" setting. We'll run this over a small data pool, but simulate as if it we were running on a "large" data pool.
For the purposes of this example, let's assume that the storage directory prefixed "s3_storage" is an s3 directory containing too much data to store locally.

## Step 1: Generate the data:
First programmatically generate some test data:
```
python examples/make_example_pools.py --target-dir test_data_inputs/fuzzy_multi --fuzzy
```

## Step 2: Build a config.yaml file:
We can do everything without a config, but it helps to store the hyperparameters in a config. Make a file with these contents and save it at `test_data_outputs/s3_storage/config.yaml`

```
# Minhash Configuration file
minhash_params:
  num_buckets: 26
  bucket_size: 11
  ngram_size: 5
  permutation_seed: 42
  tokenizer: "cl100k_base"
eng_params:
  num_docs: 1000000
  max_lines_per_path: 100000
  num_sig_chunks: 8
output_params:
  annotate: false
  annotate_key: metadata.minhash # minhash output data location
  remove_duplicates: true # just annotate, don't remove
  delete_while_cleaning: false
```

## Step 3: Generate the filemap.json.gz
We need to build a filemap object to map file indices to the actual files. This can be done either with rust:
```
cargo run --release -- mh-build-file-map --input-dir test_data_inputs/fuzzy_multi --storage-dir test_data_outputs/s3_storage/
```

Or with python
```
python python/file_map_builder.py --remote-dir test_data_inputs/fuzzy_multi --storage-dir test_data_outputs/s3_storage/
```


## Step 4: Generate the hashes
Now we generate and save hashes for all the data. Let's assume we want to do this in several slices of the data (i.e, only a slice of the corpus can fit on disk). Assuming we do this in 4 slices, call:

```
for i in {0..3}; do 
	cargo run --release -- mh-hash-docs \
		--local-input test_data_inputs/fuzzy_multi \
		--storage-dir test_data_outputs/s3_storage \
		--num-buckets 26 \
		--bucket-size 11 \
		--ngram-size 5 \
		--permutation-seed 42 \
		--path-chunk $i \
		--num-path-chunks 4
done
```
(Note: if you run into a "Too many open files" error, then just call `ulimit -n 1000` and that should fix things)

## Step 5: Generate the edges from the hashes
Now we need to generate the edges from the hashes. Instead of operating on slices of the _data_, we operate on slices in the hash space. Let's assume we do this on a band-by-band basis. To simulate this in a distributed sense, we'll copy each bands hashes to a mock "local file system" and then operate that

```

mkdir -p test_data_outputs/s3_storage/edges/

for band in test_data_outputs/s3_storage/sig_storage/*; do 
	rm -rf test_data_outputs/local_storage
	mkdir -p test_data_outputs/local_storage/sig_storage/
	cp test_data_outputs/s3_storage/filemap.json.gz test_data_outputs/local_storage
	cp -r $band test_data_outputs/local_storage/sig_storage/
	cargo run --release -- mh-gather-edges \
		--storage-dir test_data_outputs/local_storage/ 
	cp -r test_data_outputs/local_storage/edges/* test_data_outputs/s3_storage/edges/
	rm -rf test_data_outputs/local_storage
done

```


## Step 6: Generate cleaning metadata 
Then we have to run the one global step where we merge all the edges together to attain the connected components that allow us to identify duplicates.
```

cargo run --release -- mh-build-uf \
	--storage-dir test_data_outputs/s3_storage \
	--num-path-chunks 4
```



## Step 7: Scrub the output data 
And finally we have to finalize the data and apply the results of the minhash deduplication to the data. Given the `output_params` field of the config, we will annotate the data at `metadata.minhash`. In the previous step we build 4 separate files for cleaning with the `--num-path-chunks 4` argument. To simulate the nonlocality here, we will proceed by: moving each individual clean file to a directory and handle each in turn.

```
for i in {0..3}; do 
	rm -rf test_data_outputs/local_storage
	mkdir -p test_data_outputs/local_storage/clean
	cp "test_data_outputs/s3_storage/clean/chunk_0000000${i}.00000000.clean.bin" test_data_outputs/local_storage/clean/
	cp test_data_outputs/s3_storage/filemap.json.gz test_data_outputs/local_storage/
	cargo run --release -- mh-clean-files \
	--input-dir test_data_inputs/fuzzy_multi/ \
	--storage-dir test_data_outputs/local_storage \
	--output-dir test_data_outputs/fuzzy_multi \
	--path-chunk $i \
	--num-path-chunks 4 \
	--annotate false --remove-duplicates true
	rm -rf test_data_outputs/local_storage
done

```

## Step 8: Verify the outputs 
You can manually inspect the output data with snippets like: 
``` #! python
import glob, json
all_data = [json.loads(_) for f in glob.glob('test_data_outputs/fuzzy_multi/*.jsonl') for _ in open(f).read().splitlines()]
assert len(all_data) == 25 # check only 25 surviving docs
assert set(_['text'].split(' ')[-1] for _ in all_data) == set('LOREM_%02d' % i for i in range(25)) # Check the content of the surviving docs
```



