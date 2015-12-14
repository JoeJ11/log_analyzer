import os
import re

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
        return SList([(key, result[key]) for key in result])

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

    def distinct(self):
        '''
            Return a distinct element set
            Note: The order will be changed
        '''
        return SList(list(set(self)))

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
            if file_info[1] == '.shellhistory.log':
                self.log_type = LOG_TYPE_SHELL
            elif file_info[1] == '.editor.log':
                self.log_type == LOG_TYPE_EDITOR
            else:
                print "Unknown log type: {}".format(file_info[1])

    ###############################################
    ### root_path: The root path for shell logs ###
    ###############################################
    def __init__(self, root_path):
        self.NAME_PATTERN = re.compile('(.*)([0-9]{2})_([0-9]{2})_([0-9]{4})_([0-9]{2})_([0-9]{2}).log_?$')
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
