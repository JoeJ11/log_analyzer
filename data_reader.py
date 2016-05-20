import os
import re
import json
import csv
import codecs

import data_helper
from slist import SList
##################################################
### DataSet class contains all log information ###
##################################################

LOG_TYPE_TABLE = {
    '.shellhistory.log':data_helper.LogDataItem,
    '.editor.log':data_helper.EditorDataItem,
}

class DataSet:
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
            # self.item_set.append(self.DataItem(info, item))
            self.item_set.append(LOG_TYPE_TABLE[item[1]](info, item))

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
        self.user_info = {}
        for course in course_list:
            with open(os.path.join(root_path, "{}.csv".format(course)), 'r') as f_in:
                content = csv.DictReader(f_in)
                for row in content:
                    if row['email'] in self:
                        print 'User appeared in both courses: {}'.format(row['email'])
                        continue
                    self[row['email']] = course
                    self.user_info[row['email']] = row

    def get_info(self, user_name, info_name):
        return self.user_info[user_name][info_name]

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
