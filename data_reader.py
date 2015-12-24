import os
import re
import json
import csv
import codecs

##################################################
### DataSet class contains all log information ###
##################################################
LOG_TYPE_SHELL = 0
LOG_TYPE_EDITOR = LOG_TYPE_SHELL + 1

##################################
### Simulate the list of spark ###
##################################
class SList(list):
    def group_by(self, key_function):
        '''
            Similar to the semantics of spark
            Input: key
            Output: An new s-list of grouped data points based on key
        '''
        result = {}
        for item in self:
            key = key_function(item)
            if result.has_key(key):
                result[key].append(item)
            else:
                result[key] = [item]
        return SList([(key, SList(result[key])) for key in result])

    def get_element_count(self, key):
        '''
            Get a count number of distinct values for a given key
            Input: key
            Output: An integer of the number of distinct values. -1 if the key if wrong or data_set.
        '''
        if len(self) == 0:
            return 0
        if not self[0].has_key(key):
            return -1
        tem_value_list  = [item[key] for item in self]
        return len(set(tem_value_list))

    def sort_by(self, sort_func):
        '''
            Sort the dataset by given function
            Input: sorting function
            Output: A new dataset of sorted results
        '''
        return SList(sorted(self, key=sort_func))

    def count(self):
        '''
            Return the total number of elements in dataset
            Output: interger
        '''
        return len(self)

    def filter_by(self, filter_func):
        '''
            Return a filtered dataset based on filter_func
            Semantics are similar to spark
            Input: filter function
            Output: filtered dataset
        '''
        return SList(filter(filter_func, self))

    def map(self, map_func):
        '''
            Semantics are similar to spark
            Input: map function
            Output: dataset after applying map function for each item
        '''
        return SList([map_func(item) for item in self])

    def flatmap(self, map_func):
        '''
            Semantics are similar to spark
            Input: map function
            Output: dataset
        '''
        result = SList()
        for item in self:
            result += map_func(item)
        return result

    def distinct(self):
        '''
            Return a distinct element set
            Note: The order will be changed
        '''
        return SList(list(set(self)))

    def find_by(self, target_func):
        '''
            Return the first item fits target_func
            Return False if nothing fits
        '''
        for item in self:
            if target_func(item):
                return item
        return False

class DataSet:

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

            if file_info[1] == '.shellhistory.log':
                self.log_type = LOG_TYPE_SHELL
                self._parse_operation_list()
            elif file_info[1] == '.editor.log':
                self.log_type = LOG_TYPE_EDITOR
                self._parse_operation_list()
            else:
                print "Unknown log type: {}".format(file_info[1])

        def get_content(self):
            with codecs.open(self.file_path, 'r', 'utf-8') as f_in:
                return f_in.read()

        def filter_shell_input(self):
            if self.log_type != LOG_TYPE_SHELL:
                print 'Cannot apply combine operation on non-shell log!'
                return
            new_list = []
            for item in self.operation_list:
                if len(item[1]) > 1 and '27' in item[1]:
                    continue
                if len(filter(lambda x: int(x)<0, item[1])) > 0:
                    continue
                if '4' in item[1]:
                    continue
                new_list.append((item[0], item[1]))
            self.operation_list = new_list
            return self

        def combine_shell_input(self):
            if self.log_type != LOG_TYPE_SHELL:
                print 'Cannot apply combine operation on non-shell log!'
                return

            self.cmd_list = SList([])
            # self.timestamp_list = SList([])
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
                        self.cmd_list.append((self._convert_to_text(tem_cmd), int(tem_timestamp[0]), int(tem_timestamp[-1])))
                        # self.cmd_list.append(self._convert_to_text(tem_cmd))
                        tem_cmd = []
                        tem_timestamp = []
            return self

        def combien_insertion_remove(self):
            prev_command = False
            self.cmd_list = SList([])
            for item in self.operation_list:
                item['lines'][0] = u"\n".join(item['lines'])
                if not prev_command:
                    prev_command = item
                    continue
                if (prev_command['action'], item['action']) in [(u'insert', u'remove'), (u'insert', u'insert'), (u'remove', u'remove')]:
                    if item['action'] == u'insert' and prev_command['action'] == u'insert' and str(prev_command['end']) == str(item['start']):
                        prev_command['lines'][0] += item['lines'][0]
                        prev_command['end'] = item['end']
                    elif item['action'] == u'remove' and str(prev_command['start']) == str(item['end']):
                        prev_command['lines'][0] += item['lines'][0]
                        prev_command['start'] = item['start']
                    elif item['action'] == u'insert' and prev_command['action'] == u'remove' and str(prev_command['end']) == str(item['end']):
                        prev_length = len(prev_command['lines'][0])
                        tem_length = len(item['lines'][0])
                        if prev_length-tem_length<0:
                            prev_command['lines'][0] = item['lines'][0][:tem_length-prev_length]
                            prev_command['action'] = u'remove'
                        else:
                            prev_command['lines'][0] = prev_length['lines'][0][:tem_length-prev_length]
                else:
                    self.cmd_list.append(prev_command)
                    prev_command = item
            self.cmd_list.append(prev_command)
            return self

        def combine_editor_input(self):
            prev_command = False
            self.cmd_list = SList([])
            for item in self.operation_list:
                if not prev_command:
                    prev_command = item
                    continue
                if prev_command['action'] == item['action'] and item['action'] in [u'insert', u'remove']:
                    if item['action'] == u'insert' and str(prev_command['end']) == str(item['start']):
                        prev_command['lines'][0] += item['lines'][0]
                        prev_command['end'] = item['end']
                    elif item['action'] == u'remove' and str(prev_command['start']) == str(item['end']):
                        prev_command['lines'][0] += item['lines'][0]
                        prev_command['start'] = item['start']
                else:
                    self.cmd_list.append(prev_command)
                    prev_command = item
            self.cmd_list.append(prev_command)
            return self

        def _convert_to_text(self, cmd):
            text = ''
            for c in cmd:
                if int(c) in range(32,127):
                    text += chr(int(c))
                elif int(c) == 8:
                    text += ' [BACKSPACE] '
                elif int(c) == 9:
                    text += ' [TAB] '
                elif int(c) == 13:
                    text += ' [RETURN] '
                elif int(c) == 27:
                    text += ' [ESC] '
                elif int(c) == 127:
                    text += ' [DELETE]'
                else:
                    text == " [{}] ".format(c)
            return text

        def _parse_operation_list(self):
            file_content = self.get_content()
            if self.log_type == LOG_TYPE_SHELL and len(file_content) > 0:
                self._parse_shell_operation(file_content)
            elif self.log_type == LOG_TYPE_EDITOR and len(file_content) > 0:
                self._parse_editor_operation(file_content)

        def _parse_shell_operation(self, log_content):
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

        def _parse_editor_operation(self, log_content):
            lines = log_content.split(u"\n")
            for line in lines:
                if len(line) != 0:
                    self.operation_list.append(json.loads(line))

        def _strip_operation(self, item):
            if type(item).__name__ == 'str':
                return item
            else:
                return item[1]

        def filter_editor_log(self, operation_list):
            self.operation_list = filter(lambda x: x['action'] in operation_list, self.operation_list)
            return self
        # def filter_cmd_log(self, operation_list):
        #     self._cmd_list = filter(lambda x: x['action'] in operation_list, self.operation_list)
        #     return self


    ###############################################
    ### root_path: The root path for shell logs ###
    ###############################################
    def __init__(self, root_path):
        self.NAME_PATTERN = re.compile('(.*)([0-9]{2})_([0-9]{2})_([0-9]{4})_([0-9]{2})_([0-9]{2})\+?([0-9]+)?.log_?$')
        self.item_set = SList()

        os.chdir(root_path)
        self.ROOT_PATH = os.path.abspath(os.getcwd())

        for dir_item in os.listdir(self.ROOT_PATH):
            if os.path.isdir(os.path.join(self.ROOT_PATH, dir_item)):
                self._process_single_item(dir_item)
            else:
                print "Not a directory: {}".format(dir_item)

    ####################################
    ### Process single log directory ###
    ####################################
    def _process_single_item(self, dir_name):
        file_list = self._unwrap_directory(os.path.join(self.ROOT_PATH, dir_name))
        match_group = self.NAME_PATTERN.match(dir_name)
        info = [match_group.group(i+1) for i in range(match_group.lastindex)]
        for item in file_list:
            self.item_set.append(self.DataItem(info, item))

    ################################################
    ### Get all files contained in the directory ###
    ################################################
    def _unwrap_directory(self, dir_path):
        file_set = []
        for item in os.listdir(dir_path):
            tem_path = os.path.join(dir_path, item)
            if os.path.isdir(tem_path):
                file_set += self._unwrap_directory(tem_path)
            elif os.path.isfile(tem_path):
                file_set.append((tem_path, item))
            else:
                print 'ERROR::Detect an item that is neither file nor directory.'
                print "\t {}".format(tem_path)
        return file_set

####################################
### Retrieve student information ###
####################################
class StudentInformation(dict):
    def read_file(self, root_path, course_list):
        for course in course_list:
            with open(os.path.join(root_path, "{}.csv".format(course)), 'r') as f_in:
                content = csv.reader(f_in)
                for line in content:
                    if line[4] in self:
                        print 'User appeared in both courses: {}'.format(line[4])
                        continue
                    self[line[4]] = course

##############################
### Get code file template ###
##############################
class CodeTemplate(dict):
    def read_file(self, root_path, course_list, prefix='Template'):
        for course in course_list:
            with codecs.open(os.path.join(root_path, "{}_{}.txt".format(prefix, course)), 'r', 'utf-8') as f_in:
                self[course] = f_in.read()
        self._generate_line_splitter()

    def strip_template(self, content):
        content_lines = content.split(u"\n")
        content_lines = filter(lambda x: not filter(lambda y: not y in [u"\n", u"\t", u' ', u"\r"], x) in self.splitter, content_lines)
        return u"\n".join(content_lines)

    def _generate_line_splitter(self):
        self.splitter = []
        for key in self:
            for line in self[key].split("\n"):
                self.splitter.append(filter(lambda x: not x in [u"\n", u"\t", u' ', u"\r"], line))

class Anchors(dict):
    def read_file(self, root_path, course_list, prefix='Anchor'):
        for course in course_list:
            with codecs.open(os.path.join(root_path, "{}_{}.txt".format(prefix, course)), 'r', 'utf-8') as f_in:
                self[course] = f_in.read()
        self._generate_line_splitter()

    def _generate_line_splitter(self):
        self.splitter = []
        for key in self:
            for line in self[key].split("\n"):
                if len(line) > 0:
                    self.splitter.append(line)

    def assign_label(self, labels):
        self.labels = labels
        with open('anchor_label.txt', 'w') as f_out:
            for item in zip(labels, self.splitter):
                f_out.write("{}\n{}\n\n".format(item[0], item[1]))
