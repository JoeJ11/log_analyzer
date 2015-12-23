import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from scipy import stats
import codecs
import matplotlib.pyplot as plt
import json
import pickle

import data_reader
import report_tools

SHINGLE_LENGTH=2
ROUND = 100

class ClusteringAnalyzer:
    def __init__(self, filtered_editor_log, data_root):
        self.data = filtered_editor_log.map(lambda x: x.filter_editor_log(['insert', 'remove', 'paste', 'copy', 'save', 'open'])).map(lambda x: x.combine_editor_input())
        self.user_info = data_reader.StudentInformation()
        self.user_info.read_file(data_root, ['Course_A', 'Course_B', 'General'])
        self.code_template = data_reader.CodeTemplate()
        self.code_template.read_file(data_root, ['Hadoop', 'Spark'])
        self.ankors = data_reader.Ankors()
        self.ankors.read_file(data_root, ['Hadoop'])
        self.cache_table = {}
        self.global_map = {}
        self.round_maps = []


    def analyze(self):
        self.set_up()
        self.clustering_ankors()
        self._dump()

    def set_up(self):
        editor_training_set = self.data.flatmap(lambda x: x.cmd_list).filter_by(lambda x: x['action']==u'paste').map(lambda x: x['text'])
        editor_training_set = editor_training_set.map(lambda x: self.code_template.strip_template(x)).flatmap(lambda x: x.split(u"\n"))
        insert_training_set = self.data.flatmap(lambda x: x.cmd_list).filter_by(lambda x: x['action']==u'insert').map(lambda x: x['lines'][0])
        init_training_data = data_reader.SList(insert_training_set + editor_training_set)
        training_data = init_training_data.map(lambda x: self._filter_line(x))
        self._generate_dictionary(training_data)
        self._generate_round_map()
        self._generate_model(training_data)
        self._output_training_result(init_training_data)

    def clustering_ankors(self):
        ankor_label = [self.predict(item) for item in self.ankors.splitter]
        self.ankors.assign_label(ankor_label)

    def predict(self, input_line):
        filtered_line = self._filter_line(input_line)
        if self.cache_table.has_key(filtered_line):
            return self.cache_table[filtered_line]
        else:
            label = self.model.predict([self._convert_feature(filtered_line)])[0]
            self.cache_table[filtered_line] = label
            return label

    def _unicode(self, c):
        if u'\u4e00' <= c <= u'\u9fff':
            return False
        try:
            c.decode('ascii')
        except UnicodeDecodeError:
            return False
        except UnicodeEncodeError:
            return False
        return True

    def _generate_dictionary(self, training_data):
        for item in training_data:
            for index in range(len(item)-SHINGLE_LENGTH):
                tem_key = item[index:index+SHINGLE_LENGTH]
                if not self.global_map.has_key(tem_key):
                    self.global_map[tem_key] = len(self.global_map)
        print 'GlobalMap Size: {}'.format(len(self.global_map))
        with codecs.open('global_map.json', 'w', 'utf-8') as f_out:
            json.dump(self.global_map, f_out, sort_keys=True, indent=4, separators=(',', ': '))

    def _generate_round_map(self):
        for i in range(ROUND):
            round_map = np.arange(len(self.global_map))
            np.random.shuffle(round_map)
            self.round_maps.append(round_map)

    def _convert_feature(self, item):
        feature = []
        for round_map in self.round_maps:
            smallest_val = len(self.global_map)
            for index in range(len(item)-SHINGLE_LENGTH):
                tem_key = item[index:index+SHINGLE_LENGTH]
                if self.global_map.has_key(tem_key):
                    if round_map[self.global_map[tem_key]] < smallest_val:
                        smallest_val = round_map[self.global_map[tem_key]]
                else:
                    print 'Global Map Key Missing: {}'.format(tem_key)
            feature.append(smallest_val)
        return feature

    def _filter_line(self, line):
        return filter(lambda c: not c in [u"\n", u"\t", u"\r", u" "] and self._unicode(c), line)

    def _generate_model(self, training_data):
        print 'TrainingData Size: {}'.format(len(training_data))
        feature_set = [self._convert_feature(item) for item in training_data]
        print 'FeatureSet: {}*{}'.format(len(feature_set[0]), len(feature_set))
        self.model = KMeans(n_clusters=400).fit(feature_set)
        print 'Model Parameters:'
        print self.model.get_params()

    def _output_training_result(self, init_training_data):
        labels = [self.predict(item) for item in init_training_data]
        result  = zip(labels, init_training_data)
        size_list = []
        cluster_list = []
        for label in set(labels):
            tmp_result = filter(lambda x: x[0]==label, result)
            if len(tmp_result) > 100:
                size_list.append(len(tmp_result))
                cluster_list.append(label)
            with codecs.open("clustering_{}.txt".format(label), 'w', 'utf-8') as f_out:
                f_out.write(u"Size of cluster: {}\n".format(len(tmp_result)))
                for item in tmp_result:
                    f_out.write(u"{}\n".format(item[1]))
        fig, ax = report_tools.prepare_plot(figsize=(20, 5))
        ind = np.arange(len(size_list))
        width = 0.5
        ax.bar(ind, size_list, width)
        ax.set_xticks(ind+width)
        ax.set_xticklabels(['C{}'.format(i) for i in cluster_list], rotation='90')
        plt.title('Cluster size')
        plt.savefig('cluster_size.png')

    def _dump(self, ):
        with open('model.p', 'wb') as f_out:
            pickle.dump(self, f_out)

    def _load(self, file_path):
        with open(file_path, 'rb') as f_in:
            self = pickle.load(f_in)
