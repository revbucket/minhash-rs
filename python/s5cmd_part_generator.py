"""
Run this utility to generate a s5cmd command file to run when wanting to generate 
parts of the file corpus to download locally
"""


import yaml
import json
import os
from smart_open import open
from urllib.parse import urlparse
import click
import boto3


def get_path_chunk_stems(file_map_json: dict, chunk_id: int, num_chunks: int):
	return [k for k, v in file_map_json['indices'].items() if (int(v) % num_chunks) == chunk_id]


def clickfree_get_s5cmd_generator(config: str, chunk_id: int, num_chunks: int):
	config_data = yaml.safe_load(open(config, 'r'))
	working_dir = config_data['working_dir']
	file_map_loc = os.path.join(working_dir, 'filemap.json.gz')
	file_map_json = json.loads(open(file_map_loc, 'rb').read())

	path_chunk_stems = get_path_chunk_stems(file_map_json, chunk_id, num_chunks)

	line_namer = lambda stem: 'cp %s %s' % (os.path.join(config_data['remote_input'], stem),
											os.path.join(config_data['local_input'], stem))

	s5cmd_file = os.path.join(working_dir, 's5cmd_downloader_%08d_%08d.txt' % (chunk_id, num_chunks))
	with open(s5cmd_file, 'w') as f:
		f.write('\n'.join([line_namer(stem) for stem in path_chunk_stems]))	

	return s5cmd_file


@click.command()
@click.option('--config', required=True, help='Path to config.yaml file')
@click.option('--chunk-id', default=0)
@click.option('--num-chunks', default=1)
def get_s5cmd_generator(config:str, chunk_id: int, num_chunks: int):
	return clickfree_get_s5cmd_generator(config, chunk_id, num_chunks)



if __name__ == '__main__':
	get_s5cmd_generator()