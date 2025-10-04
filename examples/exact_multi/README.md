# Exact Deduplication (Multi-Node)

A test case to run exact deduplication in a "Multi-node" setting. We'll run this over a small data pool, but simulate as if we were running a "large" data pool.

## Step 1: Generate test data
First programmatically generate the test data. Call
```
python examples/make_example_pools.py --target-dir test_data_inputs/exact_multi --multi
```

## Step 2: Hash all the data and group it according to hashes
Simulating a multi-node environment, we have test data in 4 subdirectories. We'll run hashing and grouping for each of these separately
```
#!bash
for i in {0..3}; do
	cargo run --release -- exact-dedup-disk-group \
		--input-dir "test_data_inputs/exact_multi/subdir_0${i}" \
		--storage-dir test_data_outputs/exact_storage/preshuffle \
		--hash-key metadata.text_hash \
		--num-bins 4
done
```
This runs a hashing and grouping operation on each data file and stores the outputs in a storage directory. Every file in the storage directory now looks like:
```test_data_outputs/exact_storage/preshuffle/chunk_XXX.YYY.ZZZ.jsonl.zst```
where 'XXX' refers to the slice of hash ranges the contained documents have. YYY is simply the index within the slice and run. And the ZZZ is a hash referring to which slice of the data that run was performed on. 

## Step 3: Reorganize the data based on hash-slices
In continuing to simulate a large-scale run, we need to shuffle the data around such that all chunks of the same hash slice live together in a single subdirectory.
```
#!bash 
for i in {0..3}; do
	mkdir -p "test_data_outputs/exact_storage/chunk_${i}"
	for f in  test_data_outputs/exact_storage/preshuffle/chunk_0000000${i}*; do
		mv $f "test_data_outputs/exact_storage/chunk_${i}/"
	done
done
```

The mental model should be that `test_data_outputs/exact_storage/preshuffle/` has "infinite" space, but we can't operate on that in a single disk. So we need to move/download the output files
such that hash slice `i` (from each of the original data slices) lives on the same file system. 

## Step 4: Prune the duplicates out of the data 
Now we can run the prune operation on each hash-slice like:
```
#!bash
for i in {0..3}; do 
	cargo run --release -- exact-dedup-disk-prune \
		--storage-dir "test_data_outputs/exact_storage/chunk_${i}" \
		--output-dir test_data_outputs/exact_multi \
		--hash-key metadata.text_hash
done
```

And also make an annotated version of the data like:
```
#!bash
for i in {0..3}; do 
	cargo run --release -- exact-dedup-disk-prune \
		--storage-dir "test_data_outputs/exact_storage/chunk_${i}" \
		--output-dir test_data_outputs/exact_multi_anno \
		--hash-key metadata.text_hash \
		--annotate-key metadata.exact_dedup
done
```

## Step 5: Verify the outputs 
Now we can look through the outputs and verify with snippets like:
```
#!python
import zstandard, glob, json
reader = lambda x : [json.loads(_) for _ in zstandard.ZstdDecompressor().stream_reader(open(x, 'rb').read()).read().splitlines()]
all_data = [_ for f in glob.glob('test_data_outputs/exact_multi/*.jsonl.zst') for _ in reader(f)]
assert len(all_data) == 25
assert set(_['text'] for _ in all_data) == set('this doc has content: %02d' % i for i in range(25)) # Check the content of the surviving docs


annotated_data = [_ for f in glob.glob('test_data_outputs/exact_multi_anno/*.jsonl.zst') for _ in reader(f)]
assert len(annotated_data) == 325 # check all data survived
assert all(_['metadata']['exact_dedup']['num_dups'] == 25 - int(_['text'].split(' ')[-1]) for _ in annotated_data) # check data is as expected
```