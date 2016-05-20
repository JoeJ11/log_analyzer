import json
import os

with open('config.json') as f_in:
	CONFIG = json.load(f_in)
	CONFIG['data_root'] = os.path.abspath(CONFIG['data_root'])
	CONFIG['data_path'] = os.path.abspath(CONFIG['data_path'])
	CONFIG['work_dir'] = os.path.abspath(CONFIG['work_dir'])

DATA_ROOT = CONFIG['data_root']	
DATA_PATH = CONFIG['data_path']
WORK_DIR = CONFIG['work_dir']