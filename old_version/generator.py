import sys
import os
import json
import re
import collections
import datetime
import matplotlib.pyplot as plt
import pickle
import numpy as np

import data_reader
import tokenizer
import slist
import gensim

LOG_TYPE_SHELL = 0
LOG_TYPE_EDITOR = LOG_TYPE_SHELL + 1
# TOKEN_PATTERN = '[\"\'].+?[\"\']|[a-zA-Z0-9\(\)\[\]]+|[\.\{\}\:\;\=\+\-\*\/]'
TOKEN_PATTERN = '[a-zA-Z0-9\_\(\)\[\]]+|[\.\{\}\:\;\=\+\-\*\/]'
# TOKEN_PATTERN = '[a-zA-Z0-9\_]+'
class Generator():
    def __init__(self, args, reload=False):
        # Set up data
        self.data = data_reader.DataSet(args['data_path'])
        self.data_set = self.data.item_set.map(lambda x: x.combine())
        # Set up user information
        self.user_info = data_reader.StudentInformation()
        self.user_info.read_file(args['data_root'], args['user_courses'])
        # Set up code template information
        self.code_template = data_reader.CodeTemplate()
        self.code_template.read_file(args['data_root'], args['code_courses'])
        # Set up anchors
        self.anchors = data_reader.Anchors()
        self.anchors.read_file(args['data_root'], args['anchor_courses'])
        # May need reload from previous saved result (training result etc.)
        if reload:
            self._load()

    def main(self):
        content_set = self.data_set.flatmap(lambda x: x.cmd_list).map(lambda x: x['content'])
        content_set = content_set.flatmap(lambda x: re.findall('[a-zA-Z0-9]+', x))
        self.NWORDS = collections.defaultdict(lambda: 1)
        for word in content_set:
            self.NWORDS[word] += 1
        self.lstm()
        return 'Hello from generator'

    def response(self, content):
        result = ''
        content = content.replace('\'', '\"')
        content = content.replace('u\"', '\"')
        log_item = json.loads(content)
        if log_item.has_key('content'):
            correction_set = []
            for item in re.findall('[a-zA-Z0-9]+', log_item['content']):
                correction = self._correct(item)
                if correction != item:
                    correction_set.append((item, correction))
            if len(correction_set) != 0:
                for pair in correction_set:
                    result += 'Possible misspelling: {} -> {}\n'.format(pair[0], pair[1])
        return result

    def lstm(self):
        self.WORDLIB = self.data_set.flatmap(lambda x: x.cmd_list).filter_by(lambda x: x['action'] in ['paste']).map(lambda x: x['content'])
        self.WORDLIB = self.WORDLIB.map(lambda content: filter(lambda x: ord(x)<128 and ord(x)>0, content))
        self.WORDLIB = self.WORDLIB.filter_by(lambda x: len(x) > 1000 and 'import java.io.IOException;' in x)
        for index, item in enumerate(self.WORDLIB):
            with open('data/code_template_{}.txt'.format(index), 'w') as f_out:
                f_out.write(item)
        self.WORDLIB = self.WORDLIB.map(lambda x: re.findall(TOKEN_PATTERN, x))

        self.WORDSET = collections.defaultdict(lambda: 1)
        for item in self.WORDLIB.flatmap(lambda x: x):
            self.WORDSET[item] += 1

        def substitue_and_filter(input_list, NUM_INFREQUENTS=30):
            infrequent_word_counter = 0
            log_table = {}
            for index, item in enumerate(input_list):
                if item in log_table:
                    input_list[index] = 'INFREQUENT_{}'.format(log_table[item])
                if self.WORDSET[item] <= 10:
                    infrequent_word_counter += 1
                    log_table[item] = infrequent_word_counter % NUM_INFREQUENTS
                    input_list[index] = 'INFREQUENT_{}'.format(infrequent_word_counter % NUM_INFREQUENTS)
            print("number of infrequent word: {}".format(infrequent_word_counter))
            return input_list

        self.WORDLIB = self.WORDLIB.flatmap(lambda x: substitue_and_filter(x))
        # self.WORDLIB = self.WORDLIB.filter_by(lambda x: self.WORDSET[x] > 10 or x in ['INFREQUENT_{}'.format(i) for i in range(30)])

        word_set = self.WORDLIB.distinct()

        print("Total number of words: {}".format(len(self.WORDLIB.distinct())))
        with open('index_to_word.pkl', 'wb') as f_out:
            pickle.dump({idx:wd for idx, wd in enumerate(word_set)}, f_out)

        wd_to_idx = {wd:idx for idx, wd in enumerate(word_set)}
        with open('word_to_index.pkl', 'wb') as f_out:
            pickle.dump(wd_to_idx, f_out)

        training_ratio = 0.8

        with open('feature_training.pkl', 'wb') as f_out:
            pickle.dump(np.array([wd_to_idx[item] for item in self.WORDLIB[:int(training_ratio*len(self.WORDLIB))]]), f_out)

        with open('feature_validation.pkl', 'wb') as f_out:
            pickle.dump(np.array([wd_to_idx[item] for item in self.WORDLIB[int(training_ratio*len(self.WORDLIB)):len(self.WORDLIB)]]), f_out)


        # with open('code_files.txt', 'wb') as f_out:
        #     f_out.write('\n\n'.join(self.WORDLIB))

        # ds = self.data_set.group_by(lambda x: x.user_name)

    def pattern_mining(self):
        template_words = []
        for item in os.listdir('templates'):
            print('Reading template {}'.format(item))
            with open('./templates/{}'.format(item), 'r') as f_in:
                template_words += re.findall('[a-zA-Z0-9]+', f_in.read())
        template_words = list(set(template_words))
        print('Length of templates: {}'.format(len(template_words)))

        self.WORDLIB = self.data_set.flatmap(lambda x: x.cmd_list).filter_by(lambda x: x['action'] in ['paste', 'insert']).map(lambda x: x['content'])
        self.WORDLIB = self.WORDLIB.map(lambda content: filter(lambda x: ord(x)<128 and ord(x)>0, content))
        self.WORDLIB = self.WORDLIB.flatmap(lambda x: re.findall('[a-zA-Z0-9]+', x))

        ds = self.data_set.sort_by(lambda x: datetime.datetime(int(x.timestamp[2]), int(x.timestamp[1]), int(x.timestamp[0]), int(x.timestamp[3]), int(x.timestamp[4])))
        ds = ds.flatmap(lambda x: [(item, x.user_name) for item in x.cmd_list]).filter_by(lambda x: x[0]['action'] in ['paste', 'insert'])
        ds = ds.flatmap(lambda x: [(item, x[1]) for item in re.findall('[a-zA-Z0-9]+', x[0]['content'])])

        frequency_map = ds.group_by(lambda x: x[0]).map(lambda x: (x[0], len(x[1])))
        frequency_map = {item[0]:item[1] for item in frequency_map}
        print("Length of unfiltered library: {}".format(frequency_map))

        ds = ds.filter_by(lambda x: not x[0] in template_words)
        frequency_map = ds.group_by(lambda x: x[0]).map(lambda x: (x[0], len(x[1])))
        frequency_map = {item[0]:item[1] for item in frequency_map}


        def extend_libs(threshold=100):
            word_length = len(self.LIB_RESULTS) + 1
            pre_lib = self.LIB_RESULTS[len(self.LIB_RESULTS)]
            tmp_lib = collections.defaultdict(lambda: 1)

    def word_vec(self):
        self.WORDLIB = self.data_set.flatmap(lambda x: x.cmd_list).filter_by(lambda x: x['action'] in ['paste']).map(lambda x: x['content'])
        self.WORDLIB = self.WORDLIB.map(lambda content: filter(lambda x: ord(x)<128 and ord(x)>0, content))
        SHORT_LIB = self.data_set.flatmap(lambda x: x.cmd_list).filter_by(lambda x: x['action'] in ['insert', 'paste']).map(lambda x: x['content'])
        SHORT_LIB = SHORT_LIB.map(lambda content: filter(lambda x: ord(x)<128 and ord(x)>0, content)).filter_by(lambda x: len(x) <= 1000)
        self.WORDLIB = self.WORDLIB.filter_by(lambda x: len(x) > 1000 and 'import java.io.IOException;' in x)
        long_set = list(set(self.WORDLIB.flatmap(lambda x: re.findall(TOKEN_PATTERN, x))))
        short_set = list(set(SHORT_LIB.flatmap(lambda x: re.findall(TOKEN_PATTERN, x))))
        for key in short_set:
            if not key in long_set:
                print(key)
        print('********************')
        for key in long_set:
            if not key in short_set:
                print(key)
        # model = gensim.models.Word2Vec(self.WORDLIB, min_count=10, workers=4)
        # model.save('word_vec_model')
        return

    def insert_info_converter(self):
        SHORT_LIB = self.data_set.flatmap(lambda x: [(item, x.user_name) for item in x.cmd_list]).filter_by(lambda x: x[0]['action'] in ['insert'])
        SHORT_LIB = SHORT_LIB.group_by(lambda x: x[1]).map(lambda x: '\n'.join([item[0]['content'] for item in x[1]])).map(lambda x: filter(lambda c: ord(c)<128 and ord(c)>0, x))
        for index, item in enumerate(SHORT_LIB):
            with open('data/insert_{}.txt'.format(index), 'w') as f_out:
                f_out.write(item)
        # SHORT_LIB = SHORT_LIB.map(lambda content: filter(lambda x: ord(x)<128 and ord(x)>0, content)).filter_by(lambda x: len(x) <= 1000)

    def _sanitize(self, content):
        content = filter(lambda x: not x in [' ', '\t', ';', '\n', '\r'], content)
        content = filter(lambda x: ord(x) < 128 and ord(x) > 0, content)
        return content

    def _known_edits2(self, word):
        return set(e2 for e1 in self._edits1(word) for e2 in self._edits1(e1) if e2 in self.NWORDS)

    def _known(self, words): return set(w for w in words if w in self.NWORDS)

    def _edits1(self, word):
        alphabet   = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes    = [a + b[1:] for a, b in splits if b]
        transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
        replaces   = [a + c + b[1:] for a, b in splits for c in alphabet if b]
        inserts    = [a + c + b     for a, b in splits for c in alphabet]
        return set(deletes + transposes + replaces + inserts)

    def _correct(self, word):
        candidates = self._known([word]) or self._known(self._edits1(word)) or self._known_edits2(word) or [word]
        return max(candidates, key=self.NWORDS.get)

tmp_path = os.path.abspath(os.getcwd())
with open('config.json','rb') as f_in:
    config = json.load(f_in)
generator = Generator(config)
# with open('generator.pkl', 'wb') as f_out:
#     pickle.dump(generator, f_out)
# generator.main()
os.chdir(tmp_path)
# generator.pattern_mining()
generator.lstm()
# generator.word_vec()
# generator.insert_info_converter()
