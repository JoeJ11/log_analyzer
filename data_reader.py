import os
import re
import json
import csv

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
            self._operation_list = []
            self.experiment = -1
            if len(file_info) > 6:
                self.experiment = info_list[-1]

            if file_info[1] == '.shellhistory.log':
                self.log_type = LOG_TYPE_SHELL
            elif file_info[1] == '.editor.log':
                self.log_type = LOG_TYPE_EDITOR
            else:
                print "Unknown log type: {}".format(file_info[1])

        def get_content(self):
            with open(self.file_path, 'r') as f_in:
                return f_in.read()

        def get_operation_list(self):
            if len(self._operation_list) > 0:
                return SList(self._operation_list)

            file_content = self.get_content()
            if self.log_type == LOG_TYPE_SHELL:
                self._parse_shell_operation(file_content)
            elif self.log_type == LOG_TYPE_EDITOR:
                self._parse_editor_operation(file_content)

            return SList(self._operation_list)

        def combine_shell_input(self):
            if self.log_type != LOG_TYPE_SHELL:
                print 'Cannot apply combine operation on non-shell log!'
                return

            CHARACTER = [8, 9] + range(32,128)
            OMITTED_CMD = ['27', '13', '916827', '796627', '916527', '916727', '796527', '916627', '796727', '916513', '795827']
            self.cmd_list = SList([])
            self.timestamp_list = SList([])

            tem_cmd = []
            tem_stamp = []
            for item in self._operation_list:
                op = self._strip_operation(item)
                if int(op) in CHARACTER:
                    tem_cmd.append(op)
                    if self.has_timestamp:
                        tem_stamp.append(int(item[0]))
                    continue
                tem_cmd.append(op)
                tem_stamp.append(int(item[0]))
                if not ''.join(tem_cmd) in OMITTED_CMD:
                    if self.has_timestamp:
                        self.timestamp_list.append((tem_stamp[0], item[0]))
                        starting_timestamp = item[0]
                    self.cmd_list.append(self._convert_to_text(tem_cmd))
                # self.cmd_list.append([op])
                tem_cmd = []
                tem_stamp = []
            return self

        def combine_editor_input(self):
            prev_command = False
            self.cmd_list = SList([])
            for item in self._operation_list:
                if not prev_command:
                    prev_command = item
                    continue
                if prev_command['action'] == item['action'] and item['action'] in ['insert', 'remove']:
                    if item['action'] == 'insert' and str(prev_command['end']) == str(item['start']):
                        prev_command['lines'][0] += item['lines'][0]
                        prev_command['end'] = item['end']
                    elif item['action'] == 'remove' and str(prev_command['start']) == str(item['end']):
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
                if int(c) in range(32,128):
                    text += chr(int(c))
                elif int(c) == 8:
                    text += ' [DELETE] '
                elif int(c) == 9:
                    text += ' [TAB] '
                elif int(c) == 13:
                    text += ' [RETURN] '
                elif int(c) == 27:
                    text += ' [ESC] '
                else:
                    text == ' [c] '
            return text

        def _parse_shell_operation(self, log_content):
            self._operation_list = re.findall(re.compile('[0-9]{10} [0-9]+'), log_content)
            if len(self._operation_list) == 0:
                self.has_timestamp = False
                self._operation_list = re.findall(re.compile('[0-9]+'), log_content)
            else:
                self.has_timestamp = True
                self._operation_list = [item.split(' ') for item in self._operation_list]

        def _parse_editor_operation(self, log_content):
            lines = log_content.split("\n")
            for line in lines:
                if len(line) != 0:
                    self._operation_list.append(json.loads(line))

        def _strip_operation(self, item):
            if type(item).__name__ == 'str':
                return item
            else:
                return item[1]

        def filter_editor_log(self, operation_list):
            self._operation_list = filter(lambda x: x['action'] in operation_list, self._operation_list)
            return self

        # def filter_cmd_log(self, operation_list):
        #     self._cmd_list = filter(lambda x: x['action'] in operation_list, self._operation_list)
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
