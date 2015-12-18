import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats

import data_reader
import report_tools


def overall_frequency(filtered_shell_log):
    shell_input_list = _generate_counter_list(filtered_shell_log).sort_by(lambda x: int(x[0]))
    print shell_input_list.sort_by(lambda x: x[1])

    fig, ax = report_tools.prepare_plot(figsize=(20, 5), gridWidth=0.5)
    data_x = [item[0] for item in shell_input_list]
    data_y = [item[1] for item in shell_input_list]
    ind = np.arange(len(data_x))
    ax.bar(ind, data_y, 0.5)
    ax.set_xticks(ind+0.5)
    ax.set_xticklabels(data_x, rotation=70)
    plt.xlabel('Input ASCII')
    plt.ylabel('Total number of actions')
    plt.title('Frequency distribution of user inputs.')
    plt.savefig('overall_frequency.png')

def student_frequency(filtered_shell_log):
    def get_sum(counter_list):
        tmp_sum = 0
        for item in counter_list:
            tmp_sum += item[1]
        return tmp_sum
    student_input_list = filtered_shell_log.group_by(lambda x: x.user_name)
    student_input_list = student_input_list.map(lambda x: (x[0], _generate_counter_list(x[1])))
    sum_list = student_input_list.map(lambda x: (x[0], get_sum(x[1])))
    plot_data = [item[1] for item in sum_list]
    fig, ax = report_tools.prepare_plot(gridWidth = 0.5)
    ax.hist(plot_data, 50)
    plt.xlabel('Total number of actions')
    plt.ylabel('Number of students')
    plt.title('Histogram on number of actions per student')
    # plt.show()
    plt.savefig('histogram_student_total_actions.png')

def student_freq_clustering(filtered_shell_log, user_info):
    student_input_list = filtered_shell_log.group_by(lambda x: x.user_name).filter_by(lambda x: x[0] in user_info)
    student_input_list = student_input_list.map(lambda x: (x[0], _generate_counter_list(x[1])))
    freq_list = student_input_list.map(lambda x: (x[0], _convert_freq(x[1])))
    feature_list = freq_list.map(lambda x: _generate_feature_vector(x[1]))
    pca = PCA(n_components=2)
    pca.fit(feature_list)
    plot_data = pca.transform(feature_list)
    colors = []
    for item in student_input_list:
        if user_info[item[0]] == 'Course_A':
            colors.append('b')
        elif user_info[item[0]] == 'Course_B':
            colors.append('y')
        else:
            colors.append('r')
    plot_x = [item[0] for item in plot_data]
    plot_y = [item[1] for item in plot_data]

    fig, ax = report_tools.prepare_plot(gridWidth=0.5)
    plt.scatter(plot_x, plot_y, c=colors)
    plt.savefig('pca_result.png')

def student_2_means(filtered_shell_log, user_info):
    student_input_list = filtered_shell_log.group_by(lambda x: x.user_name).filter_by(lambda x: x[0] in user_info)
    student_input_list = student_input_list.map(lambda x: (x[0], _generate_counter_list(x[1])))
    freq_list = student_input_list.map(lambda x: (x[0], _convert_freq(x[1])))
    feature_list = freq_list.map(lambda x: _generate_feature_vector(x[1]))
    label_pred = KMeans(n_clusters=2).fit_predict(feature_list)

    error = 0
    for index, item in enumerate(label_pred):
        label = user_info[student_input_list[index][0]]
        if label == 'Course_A' and item == 0:
            error += 1
        if label == 'Course_B' and item == 1:
            error += 1

    rate = float(error) / float(student_input_list.count())
    if rate < 0.5:
        rate = 1 - rate
    print '2-Means Error Rate: {:.2f}'.format(rate)

def freq_statistics(filtered_shell_log, user_info):
    student_input_list = filtered_shell_log.group_by(lambda x: x.user_name).filter_by(lambda x: x[0] in user_info)
    student_input_list = student_input_list.map(lambda x: (x[0], _generate_counter_list(x[1])))
    freq_list = student_input_list.map(lambda x: (x[0], _convert_freq(x[1])))
    feature_list = freq_list.map(lambda x: _generate_feature_vector(x[1]))
    for index in range(len(feature_list[0])):
        stat, crit_vals, sig_level = stats.anderson([item[index] for item in feature_list])
        print 'INPUT {}'.format(index)
        print '\t Statistics: {}'.format(stat)
        print '\t Critical values: {}'.format(crit_vals)
        print '\t Significant levels: {}'.format(sig_level)

def cmd_counting(cmd_list):
    tmp_data = cmd_list.flatmap(lambda x: x.cmd_list).map(lambda x: str(x))
    tmp_data = tmp_data.group_by(lambda x: x).map(lambda x: (x[0], len(x[1])))
    pre_counter = tmp_data.count()
    tmp_data = tmp_data.filter_by(lambda x: x[1] < 1000 and x[1] > 5)
    post_counter = tmp_data.count()
    print "Number of filtered commands/ Total number of commands: {}/{}".format(post_counter, pre_counter)
    tmp_data = tmp_data.sort_by(lambda x: -x[1])
    print tmp_data[:30]
    plot_x = [item[0] for item in tmp_data]
    plot_y = [item[1] for item in tmp_data]
    fig, ax = report_tools.prepare_plot(gridWidth=0.5)
    ax.hist(plot_y, 50)
    plt.savefig('histogram_cmd_counter.png')

def cmd_basics(cmd_data):
    tmp_data = cmd_data.map(lambda x: len(x.cmd_list))
    tmp_data = tmp_data.filter_by(lambda x: x > 5)
    print "Total number of sessions (with threshold 5 commands): {}".format(tmp_data.count())

    fig, ax = report_tools.prepare_plot()
    ax.hist(tmp_data, 50)
    plt.savefig('histogram_cmd_per_student.png')

    tmp_data = cmd_data.filter_by(lambda x: len(x.cmd_list) > 5)
    tmp_data = tmp_data.group_by(lambda x: x.user_name).map(lambda x: len(x[1]))
    fig, ax = report_tools.prepare_plot()
    ax.hist(tmp_data, 50)
    plt.savefig('histogram_sessions_per_student.png')


    example = cmd_data.filter_by(lambda x: len(x.cmd_list) > 100).sort_by(lambda x: len(x.cmd_list))[1]
    print example.file_path
    with open('example_2.txt', 'w') as f_out:
        if example.has_timestamp:
            cmd_with_timestamp = zip(example.timestamp_list, example.cmd_list)
            for item in cmd_with_timestamp:
                f_out.write("{}-{}: {}\n".format(item[0][0], item[0][1], item[1]))
        else:
            f_out.write("\n".join(example.cmd_list))
    example = cmd_data.filter_by(lambda x: len(x.cmd_list) > 200).sort_by(lambda x: len(x.cmd_list))[1]
    print example.file_path
    with open('example_3.txt', 'w') as f_out:
        if example.has_timestamp:
            cmd_with_timestamp = zip(example.timestamp_list, example.cmd_list)
            for item in cmd_with_timestamp:
                f_out.write("{}-{}: {}\n".format(item[0][0], item[0][1], item[1]))
        else:
            f_out.write("\n".join(example.cmd_list))

def editor_basics(filtered_editor_log):
    ACTION_LIST = ['insert', 'paste', 'remove', 'copy']
    filtered_editor_log = filtered_editor_log.map(lambda x: x.filter_editor_log(ACTION_LIST))

    tmp_data = filtered_editor_log.map(lambda x: len(x._operation_list))
    fig, ax = report_tools.prepare_plot(gridWidth=0.5)
    ax.hist(tmp_data, 50)
    plt.savefig('histogram_editor_operations_per_session.png')

    print filtered_editor_log.flatmap(lambda x: x.get_operation_list()).filter_by(lambda x: x['action']=='insert')[0]
    print filtered_editor_log.flatmap(lambda x: x.get_operation_list()).filter_by(lambda x: x['action']=='paste')[0]
    print filtered_editor_log.flatmap(lambda x: x.get_operation_list()).filter_by(lambda x: x['action']=='remove')[0]
    print filtered_editor_log.flatmap(lambda x: x.get_operation_list()).filter_by(lambda x: x['action']=='copy')[0]

def _generate_counter_list(data_list):
    def _strip_operation(item):
        if type(item).__name__ == 'str':
            return item
        else:
            return item[1]
    tmp_list = data_list.flatmap(lambda x: x.get_operation_list()).map(lambda x: _strip_operation(x))
    tmp_list = tmp_list.group_by(lambda x: x).map(lambda x: (x[0], len(x[1])))
    return tmp_list

def _generate_feature_vector(counter_list):
    result = [0]*129
    for item in counter_list:
        result[int(item[0])] = item[1]
    return np.array(result)

def _convert_freq(counter_list):
    tmp_sum = 0
    for item in counter_list:
        tmp_sum += item[1]
    return data_reader.SList([(item[0], float(item[1])/float(tmp_sum)) for item in counter_list])
