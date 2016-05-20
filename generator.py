import sys
import os
import json
import datetime
import numpy as np

import config
import data_reader
import slist

LOG_TYPE_SHELL = 0
LOG_TYPE_EDITOR = LOG_TYPE_SHELL + 1
class Generator():
    def __init__(self, reload=False):
        # Set up data
        self.data = data_reader.DataSet(config.CONFIG['data_path'])
        self.data_set = self.data.item_set.map(lambda x: x.combine())
        # Set up user information
        self.user_info = data_reader.StudentInformation()
        self.user_info.read_file(config.CONFIG['data_root'], config.CONFIG['user_courses'])
        # Set up code template information
        self.code_template = data_reader.CodeTemplate()
        self.code_template.read_file(config.CONFIG['data_root'], config.CONFIG['code_courses'])
        # May need reload from previous saved result (training result etc.)
        if reload:
            self._reload()

    def _reload(self):
        return