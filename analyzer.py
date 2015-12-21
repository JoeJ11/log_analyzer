import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from scipy import stats
import codecs
import matplotlib.pyplot as plt

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
    tmp_data = cmd_list.flatmap(lambda x: x.cmd_list).map(lambda x: x[0])
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
    def _strip_input(op_list):
        result = []
        for item in op_list:
            for op in item[1]:
                result.append(op)
        return result
    tmp_list = data_list.flatmap(lambda x: _strip_input(x.operation_list))
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

def get_example():
    example = cmd_data.filter_by(lambda x: x.has_timestamp and len(x.cmd_list) > 50).sort_by(lambda x: len(x.cmd_list))[0]
    example_ = editor_cmd_data.find_by(lambda x: x.user_name == example.user_name and str(x.timestamp) == str(example.timestamp))
    combined_output = report_tools.print_log_item_with_time(example) + report_tools.print_log_item_with_time(example_)
    combined_output = sorted(combined_output, key=lambda x: x[1])
    print example.file_path
    with open('example_1_shell.html', 'w') as f_out:
        f_out.write(report_tools.print_log_item(example))
    with open('example_1_editor.html', 'w') as f_out:
        f_out.write(report_tools.print_log_item(example_).encode('utf8'))
    with open('example_1.html', 'w') as f_out:
        for item in combined_output:
            f_out.write(item[0].encode('utf8'))
        # f_out.write("\n".join([item[0] for item in combined_output]).encode('utf8'))

    example = cmd_data.filter_by(lambda x: len(x.cmd_list) > 50).sort_by(lambda x: len(x.cmd_list))[1]
    example_ = editor_cmd_data.find_by(lambda x: x.user_name == example.user_name and str(x.timestamp) == str(example.timestamp))
    print example.file_path
    with open('example_2_shell.html', 'w') as f_out:
        f_out.write(report_tools.print_log_item(example))
    with open('example_2_editor.html', 'w') as f_out:
        f_out.write(report_tools.print_log_item(example_).encode('utf8'))


    example = cmd_data.filter_by(lambda x: x.timestamp[0]=='02' and x.timestamp[1]=='12' and x.timestamp[2]=='2015' and x.user_name == 'wqf15@mails.tsinghua.edu.cn')
    print [item.timestamp for item in example]

    example_1 = example[0]
    example_2 = editor_cmd_data.find_by(lambda x: x.user_name == example_1.user_name and str(x.timestamp) == str(example_1.timestamp))
    with open('course_shell.html', 'w') as f_out:
        f_out.write(report_tools.print_log_item(example_1))
    if example_2:
        with open('course_editor.html', 'w') as f_out:
            f_out.write(report_tools.print_log_item(example_2))


    example = cmd_data.filter_by(lambda x: x.timestamp[0]=='02' and x.timestamp[1]=='12' and x.timestamp[2]=='2015' and x.user_name == 'wei.xu.0@gmail.com')
    print [item.timestamp for item in example]

    example_1 = example[1]
    example_2 = editor_cmd_data.find_by(lambda x: x.user_name == example_1.user_name and str(x.timestamp) == str(example_1.timestamp))
    with open('wei_shell.html', 'w') as f_out:
        f_out.write(report_tools.print_log_item(example_1))
    if example_2:
        with open('wei_editor.html', 'w') as f_out:
            f_out.write(report_tools.print_log_item(example_2))

    example = cmd_data.filter_by(lambda x: x.timestamp[0]=='02' and x.timestamp[1]=='12' and x.timestamp[2]=='2015' and x.timestamp[3] in ['03', '04'])

    for index, item in enumerate(example):
        print item.user_name
        with open('all_{}.html'.format(index), 'w') as f_out:
            f_out.write(report_tools.print_log_item(item))

def editor_behavior_analysis(filtered_editor_log, code_template, user_info):
    def get_operation_history(session_history):
        result = []
        for session in session_history:
            result += session.cmd_list
        return result
    def split_history(cmd_history, op_list):
        tmp_session = []
        result = []
        for item in cmd_history:
            tmp_session.append(item)
            if item['action'] in op_list:
                result.append(tmp_session)
                tmp_session = []
        return filter(lambda x: len(x) > 0, result)

    editor_cmd_data = filtered_editor_log.map(lambda x: x.filter_editor_log([u'insert', u'remove', u'paste', u'copy', u'save', u'open'])).map(lambda x: x.combine_editor_input())

    # plot_data = editor_cmd_data.flatmap(lambda x: x.cmd_list).filter_by(lambda x: x['action']=='paste').map(lambda x: x['text'])
    # with codecs.open('middle_paste.txt', 'w', 'utf-8') as f_out:
    #     f_out.write("\n***************************\n".join(plot_data.filter_by(lambda x: len(x)>1000 and len(x)<3000)))
    # with codecs.open('long_paste.txt', 'w', 'utf-8') as f_out:
    #     f_out.write("\n***************************\n".join(plot_data.filter_by(lambda x: len(x)>4000)))
    # with codecs.open('short_paste.txt', 'w', 'utf-8') as f_out:
    #     f_out.write("\n***************************\n".join(plot_data.filter_by(lambda x: len(x)<1000)))
    template_filtered_data = editor_cmd_data.flatmap(lambda x: x.cmd_list).filter_by(lambda x: x['action']==u'paste').map(lambda x: x['text'])
    template_filtered_data = template_filtered_data.map(lambda x: code_template.strip_template(x))
    fig, ax = report_tools.prepare_plot(gridWidth=0.5)
    ax.hist(template_filtered_data.map(lambda x: len(x)).filter_by(lambda x: x<1000), 50)
    plt.title('Histogram on filtered pasted content length')
    plt.savefig('hist_filtered_pasted_content.png')

    with codecs.open('middle_filtered_paste.txt', 'w', 'utf-8') as f_out:
        f_out.write("\n***************************\n".join(template_filtered_data.filter_by(lambda x: len(x)>1000 and len(x)<3000)))
    with codecs.open('long_filtered_paste.txt', 'w', 'utf-8') as f_out:
        f_out.write("\n***************************\n".join(template_filtered_data.filter_by(lambda x: len(x)>4000)))
    with codecs.open('short_filtered_paste.txt', 'w', 'utf-8') as f_out:
        f_out.write("\n***************************\n".join(template_filtered_data.filter_by(lambda x: len(x)<1000)))


    # fig, ax = report_tools.prepare_plot(gridWidth=0.5)
    # ax.hist(plot_data.map(lambda x: len(x)).filter_by(lambda x: x<1000), 50)
     #plt.title('Histogram on length of pasted contents')
    # plt.xlabel('Length of pasted content')
    # plt.savefig('hist_length_pasted_content.png')

    student_history_data = editor_cmd_data.group_by(lambda x: x.user_name).map(lambda x: (x[0], get_operation_history(x[1])))
    tmp_data = student_history_data.map(lambda x: (x[0], split_history(x[1], ['save', 'open'])))

    fig, ax = report_tools.prepare_plot(gridWidth=0.5)
    ax.hist(tmp_data.map(lambda x: len(x[1])), 50)
    plt.title('Histogram on number of editor sessions')
    plt.xlabel('Number of editor sessions')
    plt.savefig('hist_editor_session.png')

    tmp_data = student_history_data.map(lambda x: filter(lambda y: y['action']=='paste', x[1])).map(lambda x: "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n".join([item['text'] for item in x]))
    counting_data = tmp_data.group_by(lambda x: x).map(lambda x: (x[0], len(x[1]))).sort_by(lambda x: -x[1])
    with codecs.open('pasted_content.txt', 'w', 'utf-8') as f_out:
        for item in counting_data:
            f_out.write(str(item[1]))
            f_out.write("\n\n")
            f_out.write(item[0])
            f_out.write("\n****************************************************************************\n")
    fig, ax = report_tools.prepare_plot(gridWidth=0.5)
    ax.hist(tmp_data.map(lambda x: len(x)), 50)
    plt.title('Histogram on length of pasted content')
    plt.xlabel('Length of pasted content')
    plt.savefig('hist_pasted_content.png')

def editor_input_clustering(filtered_editor_log, code_template, user_info):
    editor_cmd_data = filtered_editor_log.map(lambda x: x.filter_editor_log(['insert', 'remove', 'paste', 'copy', 'save', 'open'])).map(lambda x: x.combine_editor_input())
    insert_data = editor_cmd_data.flatmap(lambda x: x.cmd_list).filter_by(lambda x: x['action']==u'insert').map(lambda x: x['lines'][0])
    template_filtered_data = editor_cmd_data.flatmap(lambda x: x.cmd_list).filter_by(lambda x: x['action']==u'paste').map(lambda x: x['text'])
    template_filtered_data = template_filtered_data.map(lambda x: code_template.strip_template(x))
    total_input = data_reader.SList(insert_data + template_filtered_data.flatmap(lambda x: x.split(u"\n")))
    total_input = total_input.filter_by(lambda x: len(x)>0)
    print len(total_input)
    feature_set = _generate_feature_set(total_input)
    print len(feature_set)
    # pca = PCA(n_components=2)
    # pca.fit(feature_set)
    # plot_data = pca.transform(feature_set)

    # fig, ax = report_tools.prepare_plot()
    # ax.scatter([item[0] for item in plot_data], [item[1] for item in plot_data])
    # plt.title('Scatter plot on editor input')
    # plt.savefig('scatter_editor_input.png')

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter([item[0] for item in plot_data], [item[1] for item in plot_data], [item[2] for item in plot_data])
    # plt.title('Scatter plot on editor input')
    # plt.savefig('3d_scatter_editor_input.png')

    db = DBSCAN(eps=1000, min_samples=100).fit(feature_set)
    labels = db.labels_
    result  = zip(labels, total_input)
    print len(set(labels))
    for label in len(set(labels)):
        with codecs.open("clustering_{}.txt".format(label), 'r', 'utf-8') as f_out:
            tmp_result = filter(lambda x: x[0]==label, result)
            f_out.write("Size of cluster: {}\n".format(len(tmp_result)))
            for item in tmp_result:
                f_out.write("{}\n".format(item[1]))

def _generate_feature_set(editor_input):
    editor_input = [filter(lambda x: x!="\t" and x!="\r", x) for x in editor_input]
    SHINGLE_LENGTH=2
    ROUND = 100
    global_map = {}
    def _generate_shingle_dict(input_line):
        for index in range(len(input_line)-SHINGLE_LENGTH):
            tem_key = input_line[index:index+SHINGLE_LENGTH]
            if not global_map.has_key(tem_key):
                global_map[tem_key] = len(global_map)
    def _generate_feature(all_input):
        round_map = np.arange(len(global_map))
        result = [[] for i in range(len(all_input))]
        for rd in range(ROUND):
            np.random.shuffle(round_map)
            for tem_index, item in enumerate(all_input):
                smallest_val = len(global_map)
                for index in range(len(item)-SHINGLE_LENGTH):
                    tem_key = item[index:index+SHINGLE_LENGTH]
                    if round_map[global_map[tem_key]] < smallest_val:
                        smallest_val = round_map[global_map[tem_key]]
                result[tem_index].append(smallest_val)
        return result

    for item in editor_input:
        _generate_shingle_dict(item)
    print len(global_map)
    return _generate_feature(editor_input)
