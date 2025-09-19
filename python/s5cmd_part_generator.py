"""
Run this utility to generate a s5cmd command file to run when wanting to generate 
parts of the file corpus to download locally
"""


import gzip
import json
import os
from urllib.parse import urlparse

import boto3
import click
import yaml
from smart_open import open


def get_path_chunk_stems(filemap_json: dict, chunk_id: int, num_chunks: int):
    return [
        k
        for k, v in filemap_json["indices"].items()
        if (int(v) % num_chunks) == chunk_id
    ]


def clickfree_get_s5cmd_generator(
    filemap: str, local_dir: str, output_dir: str, chunk_id: int, num_chunks: int
):
    # Read filemap
    if filemap.startswith("s3://"):
        s3_parts = filemap.replace("s3://", "").split("/", 1)
        bucket_name = s3_parts[0]
        key = s3_parts[1]
        filemap_response = boto3.client("s3").get_object(Bucket=bucket_name, Key=key)
        filemap_content = gzip.decompress(filemap_response["Body"].read())
    else:
        filemap_content = open(filemap, "rb").read()
    filemap_json = json.loads(filemap_content)

    # Make output string
    path_chunk_stems = get_path_chunk_stems(filemap_json, chunk_id, num_chunks)
    line_namer = lambda stem: "cp %s %s" % (
        os.path.join(filemap_json["remote_input"], stem),
        os.path.join(local_dir, stem),
    )

    s5cmd_file = os.path.join(
        output_dir, "s5cmd_downloader_%08d_%08d.txt" % (chunk_id, num_chunks)
    )
    print(
        "S%C",
        s5cmd_file,
        len("\n".join([line_namer(stem) for stem in path_chunk_stems])),
    )
    output_str = "\n".join([line_namer(stem) for stem in path_chunk_stems])

    # Save output string
    if s5cmd_file.startswith("s3://"):
        s3_parts = s5cmd_file.replace("s3://", "").split("/", 1)
        bucket_name = s3_parts[0]
        key = s3_parts[1]
        s3_client = boto3.client("s3")
        s3_client.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=output_str.encode("utf-8"),  # Convert string to bytes
            ContentType="text/plain",
        )
    else:
        os.makedirs(output_dir, exist_ok=True)
        with open(s5cmd_file, "w") as f:
            f.write(output_str)

    return s5cmd_file


@click.command()
@click.option(
    "--filemap", required=True, help="Path to filemap.json.gz (local or on s3)"
)
@click.option(
    "--local-dir",
    required=True,
    help="Path to where the data should live once downloaded",
)
@click.option(
    "--output-dir",
    required=True,
    help="Where the download file lives -- (could be on s3)",
)
@click.option("--chunk-id", default=0)
@click.option("--num-chunks", default=1)
def get_s5cmd_generator(
    filemap: str, local_dir: str, output_dir: str, chunk_id: int, num_chunks: int
):
    return clickfree_get_s5cmd_generator(
        filemap, local_dir, output_dir, chunk_id, num_chunks
    )


if __name__ == "__main__":
    get_s5cmd_generator()
