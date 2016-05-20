import os
import re
import json
import csv
import codecs
from slist import SList

LOG_TYPE_SHELL = 0
LOG_TYPE_EDITOR = LOG_TYPE_SHELL + 1

##########################
### A single data item ###
##########################
class DataItem:
    def __init__(self, info_list, file_info):
        self.user_name = info_list[0]
        self.timestamp = info_list[1:6]
        self.file_path = file_info[0]
        self.log_type = -1
        self.operation_list = []
        self.experiment = -1
        if len(file_info) > 6:
            self.experiment = info_list[-1]
        self._parse_operation_list()

    def get_content(self):
        with codecs.open(self.file_path, 'r', 'utf-8') as f_in:
            return f_in.read()

    def combine(self):
        return self

    def _parse_operation_list(self):
        file_content = self.get_content()
        if len(file_content) > 0:
            self._parse_operation(file_content)

    def _strip_operation(self, item):
        if type(item).__name__ == 'str':
            return item
        else:
            return item[1]

class LogDataItem(DataItem):
    def __init__(self, info_list, file_info):
        DataItem.__init__(self, info_list, file_info)
        self.log_type = LOG_TYPE_SHELL

    def _parse_operation(self, log_content):
        self.operation_list = re.findall(re.compile('[0-9]{10} [0-9]+'), log_content)
        if len(self.operation_list) == 0:
            self.has_timestamp = False
        else:
            self.has_timestamp = True
        for line in log_content.split("\n"):
            tmp_list = filter(None, line.split(' '))
            if len(tmp_list) > 1 and self.has_timestamp:
                tmp_timestamp = tmp_list[0]
                self.operation_list.append((tmp_timestamp, filter(lambda x: x!=tmp_timestamp, tmp_list)))
            elif len(tmp_list) > 0 and not self.has_timestamp:
                self.operation_list.append(('0', tmp_list))

    def combine(self):
        self.cmd_list = SList([])
        CHARACTER = [8, 9] + range(32,128)
        tem_cmd = []
        tem_timestamp = []
        for item in self.operation_list:
            if len(item[1]) > 1 and '27' in item[1]:
                continue
            if len(filter(lambda x: int(x)<0, item[1])) > 0:
                continue
            if '4' in item[1]:
                continue
            for op in item[1]:
                tem_timestamp.append(item[0])
                tem_cmd.append(op)
                if not int(op) in CHARACTER:
                    content, delimiter = self._convert_to_text(tem_cmd)
                    self.cmd_list.append({
                        'action':'shell',
                        'content':content,
                        'delimiter':delimiter,
                        'timestamp':int(tem_timestamp[0]),
                        'timestamp_end':int(tem_timestamp[-1])
                    })
                    tem_cmd = []
                    tem_timestamp = []
        return self

    def _convert_to_text(self, cmd):
        text = ''
        rtn = ''
        for c in cmd:
            if int(c) in range(32,127):
                text += chr(int(c))
            elif int(c) == 8:
                text = text[:-1]
            elif int(c) == 9:
                text += '[TAB]'
            elif int(c) == 13:
                rtn = '[RETURN]'
            elif int(c) == 27:
                rtn = '[ESC]'
            elif int(c) == 127:
                rtn = '[DELETE]'
            else:
                rtn = "[{}]".format(c)
        return text, rtn

class EditorDataItem(DataItem):
    def __init__(self, info_list, file_info):
        DataItem.__init__(self, info_list, file_info)
        self.log_type = LOG_TYPE_EDITOR

    def _parse_operation(self, log_content):
        lines = log_content.split(u"\n")
        for line in lines:
            if len(line) != 0:
                self.operation_list.append(json.loads(line))

    def combine(self):
        prev_command = False
        self.cmd_list = SList([])
        for item in self.operation_list:
            if item['action'] in ['insert','remove']:
                item['content'] = u"\n".join(item['lines'])
            elif item['action'] in ['copy', 'paste']:
                item['content'] = item['text']
            elif item['action'] in ['open','save']:
                item['content'] = ''
            else:
                continue

            if not prev_command:
                prev_command = item
                continue
            if (prev_command['action'], item['action']) in [(u'insert', u'remove'), (u'insert', u'insert'), (u'remove', u'remove')]:
                if item['action'] == u'insert' and prev_command['action'] == u'insert' and str(prev_command['end']) == str(item['start']):
                    prev_command['content'] += item['content']
                    prev_command['end'] = item['end']
                elif item['action'] == u'remove' and str(prev_command['start']) == str(item['end']):
                    prev_command['content'] += item['content']
                    prev_command['start'] = item['start']
                elif item['action'] == u'insert' and prev_command['action'] == u'remove' and str(prev_command['end']) == str(item['end']):
                    prev_length = len(prev_command['content'])
                    tem_length = len(item['content'])
                    if prev_length-tem_length<0:
                        prev_command['content'] = item['content'][:tem_length-prev_length]
                        prev_command['action'] = u'remove'
                    else:
                        prev_command['content'] = prev_length['content'][:tem_length-prev_length]
                else:
                    self.cmd_list.append(prev_command)
                    prev_command = item
            else:
                self.cmd_list.append(prev_command)
                prev_command = item
        if prev_command:
            self.cmd_list.append(prev_command)
        return self
