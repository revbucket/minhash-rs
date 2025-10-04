# Fuzzy Deduplication (Single-Node)

A test case to run fuzzy dedudplication in a single-node setting.

## Step 1: Generate test data
First programmatically generate some test data. Call 
```
python examples/make_example_pools.py --target-dir test_data_inputs/fuzzy --fuzzy
```

## Step 2: Run the fuzzy deduplication 


Call the rust tool to generate a deduplicated data pool. We can use a config to stash all these parameters, but we'll be explicit with our parameters here. To deduplicate and emit only the "unique" documents, run 
```
cargo run --release -- minhash-memory \
	--input-dir test_data_inputs/fuzzy \
	--storage-dir test_data_outputs/fuzzy_mem_storage \
	--output-dir test_data_outputs/fuzzy_mem \
	--num-buckets 26 \
	--bucket-size 11 \
	--ngram-size 5 \
	--permutation-seed 42 \
	--cleanup-storage
```

Or to annotate the data: 
```
cargo run --release -- minhash-memory \
	--input-dir test_data_inputs/fuzzy \
	--storage-dir test_data_outputs/fuzzy_mem_storage \
	--output-dir test_data_outputs/fuzzy_mem_anno \
	--num-buckets 26 \
	--bucket-size 11 \
	--ngram-size 5 \
	--permutation-seed 42 \
	--cleanup-storage \
	--remove-duplicates false \
	--annotate true \
	--annotate-key metadata.minhash
```


## Step 3: Verify the outputs
You can manually inspect the output data with snippets like: 
``` #! python
import glob, json
all_data = [json.loads(_) for f in glob.glob('test_data_outputs/fuzzy_mem/*.jsonl') for _ in open(f).read().splitlines()]
assert len(all_data) == 25 # check only 25 surviving docs
assert set(_['text'].split(' ')[-1] for _ in all_data) == set('LOREM_%02d' % i for i in range(25)) # Check the content of the surviving docs


annotated_data = [json.loads(_) for f in glob.glob('test_data_outputs/fuzzy_mem_anno/*.jsonl') for _ in open(f).read().splitlines()]
assert len(annotated_data) == 325 # check all data survived
missing_minhash_count = 0
for doc in annotated_data:
	if 'metadata' not in doc:
		missing_minhash_count += 1
	else:
		assert doc['metadata']['minhash']['cc_size'] == 25 - int(doc['text'].split('_')[-1])

assert missing_minhash_count == 1
```
