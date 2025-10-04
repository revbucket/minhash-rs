# Minhash Case Study: Essential-Web-v1.0

In this case study, we'll demonstrate how we use this tool to run minhash deduplication on a large-scale corpus of webdata. As an example, we'll be using Essential-Web-v1.0, which has 23.6 billion documents and 24 trillion tokens. See the [data card](https://huggingface.co/datasets/EssentialAI/essential-web-v1.0), but the salient characteristics:
- 101 resiliparse processed commoncrawl snapshots
- Exact deduplication was performed globally, but minhash deduplication was performed on a per-snapshot level with 14 bands and 9 rows per band.
- 54.8 TB of data when converted to a jsonl.zst format

Let's assume that we have stored this data in a .jsonl.zst format on S3, located at `s3://BUCKET/essential-web/data/`. Further, let's assume that we only have access to EC2 i4i.32xlarge instances, which have 1TB of RAM and a maximum of ~27TB of disk space using the AWS Nitro drives.

## Step 1a: Collect all the target files
The first step is to get a collection of all the files we wish to deduplicate. Because the whole corpus cannot fit onto one disk, we have do this by asking S3 directly using the python tooling. 

To do this, on any machine with fast internet and S3 privileges, run:
```
python python/file_map_builder.py --storage-dir s3://BUCKET/essential-web/minhash/ --remote-dir s3://BUCKET/essential-web/data/
```

This will create an object called `filemap.json.gz` and store it with location `s3://BUCKET/essential-web/minhash/filemap.json.gz`, which is just a json file with schema like:
```
{
  "remote_input": "s3://BUCKET/essential-web/data/"
  "indices": {
    "file_0.jsonl.zst": 0,
    "file_1.jsonl.zst": 1,
    ...
  }
}
```


## Step 1b: Make config
YAML configuration files are optional, but there are many specifications and it becomes tedious to feed these in to every task you call. The structure of what the config should contain is located in `/src/minhash_config.rs`, but for our cases, the example config should look like:

```
# Minhash Configuration file
minhash_params:
  num_buckets: 26
  bucket_size: 11
  ngram_size: 5
  permutation_seed: 42
  tokenizer: "cl100k_base"
eng_params:
  num_docs: 24_000_000_000 # Supposed to be 23.6B documents
  max_line_per_path: 1_048_576 # Depends on how you made the jsonl.zst's
  num_sig_chunks: 65536
output_params:
  annotate: true
  annotate_key: metadata.minhash # minhash output data location
  delete_while_cleaning: true # does housekeeping during final step
  remove_duplicates: false # just annotate, don't remove
```

Save this config next to the filemap, for example at `s3://BUCKET/essential-web/minhash/config.yaml`


## Step 1c: Download slices of the corpus
With one global lookup for which files we want to process in hand, we can start running the many hashes of the data. We can parallelize this step across _input files_. It helps to build out tooling to download slices of the corpus and process each one either sequentially or in a distributed sense. 

Suppose we want to break this into 100 slices. To maximize download speed, we use the [s5cmd](https://github.com/peak/s5cmd) tool to interact with s3. To execute an `s5cmd run` command, we need to know exactly which files to download in each slice, and when executed, the slice of the data should be stored at `$LOCAL_DATA`. To do this, run

```
for i in {0..99}; do 
    python python/s5cmd_part_generator.py \
      --filemap s3://BUCKET/essential-web/minhash/filemap.json.gz \
      --local-dir $LOCAL_DATA \
      --output-dir s3://BUCKET/essential-web/minhash/slice_download_scripts/ \
      --chunk-id $i \
      --num-chunks 100
done
```

This makes 100 files stored like `s3://BUCKET/essential-web/minhash/slice_download_scripts/s5cmd_downloader_00000001_00000100.txt` 

Then to execute one of these scripts, simply download it onto an i4i.32xlarge instance and call 

```s5cmd run s5cmd_downloader_00000001_00000100.txt```

## Step 2: Hash the data
For each of the slices of data, we need to generate hashes to be used by the minhash algorithm. The exact theory of what's going on here is discussed in the "theory" README. Up until now, everything should be fairly quick (though downloading the data may take a while). 

The next several steps will take some serious compute time, with hashing perhaps being the heaviest one. Just to be clear, at hash time, the following things are needed *on disk*:
- The filemap.json.gz (located at `$LOCAL_STORAGE/filemap.json.gz`
- The config.yaml (or a list of all the override specifications)
- A slice of data stored locally in a .jsonl.* format
- Enough storage to store the hash signatures for the data (typically about 15-25% of the original data size)

And then run the following incantation:

```cargo run --release -- mh-hash-docs --local-input $LOCAL_DATA --storage-dir $LOCAL_STORAGE --config $PATH_TO_CONFIG --path-chunk $SLICE_NUM --num-path-chunks $NUM_SLICES```

Which will populate the `$LOCAL_STORAGE/sig_storage` directory with hash signatures of the documents.


## Step 3: Gather edges
With all the hashes for all the slices complete, the next step is to "gather edges", and identify a connected graph that contains all the information about the duplicate documents. 

The parallelism here is across the bands. If you'll notice, all the signature files have file structure like
```$LOCAL_STORAGE/sig_storage/band_XXX/sigchunk_YYYY/pathchunk_ZZZZ.sig.bin```
Where this file means that this file contains the documents and signatures for the Z'th slice of data, the X'th band, and just signatures starting with Y'th slice of the signature namespace. Since we seek to "reduce" the "map" here across only a single band and sigchunk, we need only move the data around so that any one node contains, for every (XXXX, YYYY), pathchunk_Z.sig.bin files.

So at this step, we need, *on disk*:
- The filemap.json.gz (located at `$LOCAL_STORAGE/filemap.json.gz`
- The config.yaml (or a list of all the override specifications)
- All the pathchunk_ZZZ.sig.bin's for a given (set of) `(band, sigchunk)` pairs.
- Enough storage to hold the edge files: all edge files combined are about 1-10% of the original data size.

And then run the following incantation:
```cargo run --release -- mh-gather-edges --storage-dir $LOCAL_STORAGE --config $PATH_TO_CONFIG```

## Step 4: Build Union Find 
With all the edges collected, we now do the only non-parallelizable step of the minhash deduplication process where we build a union find data structure and then, for all documents that have a connected component of size > 1, we keep track of its connected-component (duplicate group) ID, size, and imbue an ordering within each connected-component. We stash this data in a file that can be used to either annotate or remove duplicates.

At this step, we need, *on disk*:
- The filemap.json.gz (located at `$LOCAL_STORAGE/filemap.json.gz`
- The config.yaml (or a list of all the override specifications)
- **All** the edge files
- Enough storage to hold the duplicate data, which is typically <1% of the original data size.


And then run the following incantation:

```cargo run --release -- mh-build-uf --storage-dir $LOCAL_STORAGE --config $PATH_TO_CONFIG  --num-path-chunks <NUM_PATH_CHUNKS>```

(Note: `<NUM_PATH_CHUNKS> refers to how we're going to distribute the next step. This can be left at 1 for all but the largest cases)


## Step 5: Clean/Annotate the data
Finally we can make a pass over the original data and either: i) remove the duplicate documents from each file; or ii) annotate the documents with their duplicate information. Once again, this can be distributed across slices of the input data.

At this step, we need, on *disk*:
- The filemap.json.gz (located at `$LOCAL_STORAGE/filemap.json.gz`
- The config.yaml (or a list of all the override specifications)
- The `.clean.bin` files generated for the appropriate slice of data from the previous step
- The slice of input data we seek to annotate. Recall that you can always redownload it onto an s3 instance using one of the download scripts generated in step 1c.

Then the incantation looks like:

``` cargo run --release -- mh-clean-files --input-dir $LOCAL_DATA --storage-dir $LOCAL_STORAGE --output-dir $OUTPUT_DIR --config $PATH_TO_CONFIG --path-chunk <CHUNK> --num-path-chunks <NUM_CHUNKS>```


Where the output will then have identical filenames to `$LOCAL_DATA`, but located in `$OUTPUT_DIR`

