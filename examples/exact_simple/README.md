# Exact Deduplication (Simple)

A test case to run exact deduplication in a single-node setting. 

## Step 1: Generate test data
First programmatically generate the test data. Call
```
python examples/make_example_pools.py --target-dir test_data_inputs/exact
```

## Step 2: Run exact deduplication
Call the rust tool to generate a deduplicated pool of data (both in an annotated and non-annotated fashion).

To clean the data:
```cargo run --release exact-dedup-memory --input-dir test_data_inputs/exact --output-dir test_data_outputs/exact_memory```
And to just annotate the data:
```cargo run --release exact-dedup-memory --input-dir test_data_inputs/exact --output-dir test_data_outputs/exact_memory_anno --annotate-key metadata.exact_dedup ```


## Step 3: Verify the outputs
You can manually inspect the output data with snippets like:

``` #! python
import glob, json
all_data = [json.loads(_) for f in glob.glob('test_data_outputs/exact_memory/*.jsonl') for _ in open(f).read().splitlines()]
assert len(all_data) == 25 # check only 25 surviving docs
assert set(_['text'] for _ in all_data) == set('this doc has content: %02d' % i for i in range(25)) # Check the content of the surviving docs


annotated_data = [json.loads(_) for f in glob.glob('test_data_outputs/exact_memory_anno/*.jsonl') for _ in open(f).read().splitlines()]
assert len(annotated_data) == 325 # check all data survived
assert all(_['metadata']['exact_dedup']['num_dups'] == 25 - int(_['text'].split(' ')[-1]) for _ in annotated_data) # check data is as expected
```

