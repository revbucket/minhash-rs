"""
Run this utility to generate the file map (helpful because I don't let rust talk to s3)
"""
import yaml
import json
import os
from smart_open import open
from urllib.parse import urlparse
import click
import boto3

def parse_s3_uri(uri):
    parsed = urlparse(uri)
    if parsed.scheme != 's3':
        raise ValueError("URI must be an S3 URI with 's3://' scheme")
        
    # Remove leading slash from path
    prefix = parsed.path.lstrip('/')
    
    return parsed.netloc, prefix


def list_s3_files(bucket_name, prefix, contains=None):
    s3_client = boto3.client('s3')
    
    # Use paginator to handle cases with more than 1000 objects
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    
    file_list = []
    for page in pages:
        if 'Contents' in page:  # Check if there are any objects
            for obj in page['Contents']:
                file_list.append('s3://%s/%s' % (bucket_name, obj['Key']))
    
    if contains != None:
        file_list = [_ for _ in file_list if contains in _]                
    return file_list



@click.command()
@click.option('--config', required=True, help='Path to config.yaml file')
def build_file_map(config: str):
    config_data = yaml.safe_load(open(config, 'r'))

    working_dir = config_data['working_dir']
    os.makedirs(working_dir, exist_ok=True)

    bucket, prefix = parse_s3_uri(config_data['remote_input'])
    files = list_s3_files(bucket, prefix, contains='.jsonl')
    file_map_loc = os.path.join(working_dir, 'filemap.json.gz')
    file_map_contents = {'local_input': config_data['local_input'],
                        'remote_input': config_data['remote_input'],
                        'indices': {p.replace(config_data['remote_input'], '') : i for i,p in enumerate(files)}}


    with open(file_map_loc, 'wb') as f:
        f.write(json.dumps(file_map_contents).encode('utf-8'))


if __name__ == '__main__':
    build_file_map()



