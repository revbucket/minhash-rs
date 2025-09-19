"""
Run this utility to generate the file map (helpful because I don't let rust talk to s3)
"""
import glob
import json
import os
from urllib.parse import urlparse

import boto3
import click
import yaml
from smart_open import open


def parse_s3_uri(uri):
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError("URI must be an S3 URI with 's3://' scheme")

    # Remove leading slash from path
    prefix = parsed.path.lstrip("/")

    return parsed.netloc, prefix


def list_s3_files(bucket_name, prefix, contains=None):
    s3_client = boto3.client("s3")

    # Use paginator to handle cases with more than 1000 objects
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    file_list = []
    for page in pages:
        if "Contents" in page:  # Check if there are any objects
            for obj in page["Contents"]:
                file_list.append("s3://%s/%s" % (bucket_name, obj["Key"]))

    if contains != None:
        file_list = [_ for _ in file_list if contains in _]
    return file_list


def clickfree_build_file_map(storage_dir, remote_dir):
    os.makedirs(storage_dir, exist_ok=True)

    if remote_dir.startswith("s3://"):
        bucket, prefix = parse_s3_uri(remote_dir)
        files = list_s3_files(bucket, prefix, contains=".jsonl")
    else:
        files = [
            filename
            for filename in glob.glob(os.path.join(remote_dir, "**/*"), recursive=True)
            if ".jsonl" in filename
        ]

    file_map_loc = os.path.join(storage_dir, "filemap.json.gz")
    file_map_contents = {
        "remote_input": remote_dir,
        "indices": {
            p.replace(remote_dir, "").lstrip("/"): i for i, p in enumerate(files)
        },
    }

    with open(file_map_loc, "wb") as f:
        f.write(json.dumps(file_map_contents).encode("utf-8"))


@click.command()
@click.option(
    "--storage-dir", required=True, help="Location where filemap.json.gz should live"
)
@click.option(
    "--remote-dir",
    required=True,
    help="Location (either s3 or local) where the data lives",
)
def build_file_map(storage_dir: str, remote_dir: str):
    return clickfree_build_file_map(storage_dir, remote_dir)


if __name__ == "__main__":
    build_file_map()
