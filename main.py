import sys
import numpy as np
import matplotlib.pyplot as plt
import os

import data_reader
import report_tools
import analyzer

data_path = sys.argv[1]
output_path = os.path.abspath(os.path.join(os.getcwd(), sys.argv[2]))

student_file_root = os.path.abspath(os.path.join(data_path, '..'))
user_info = data_reader.StudentInformation()
user_info.read_file(student_file_root, ['Course_A', 'Course_B', 'General'])

data = data_reader.DataSet(data_path)
data_set = data.item_set
editor_log = data_set.filter_by(lambda x: x.log_type == data_reader.LOG_TYPE_EDITOR)
shell_log = data_set.filter_by(lambda x: x.log_type == data_reader.LOG_TYPE_SHELL)

os.chdir(output_path)
print 'LENGTH: {}'.format(data_set.count())
print 'NUMBER of STUDENTS: {}'.format(data_set.group_by(lambda x: x.user_name).count())
print 'Number of shell log: {}'.format(shell_log.count())
print 'Number of editor log: {}'.format(editor_log.count())

filtered_editor_log = editor_log.filter_by(lambda x: len(x.get_operation_list())!=0)
filtered_shell_log = shell_log.filter_by(lambda x: len(x.get_operation_list())!=0)
print 'Number of Non-empty shell log: {}'.format(filtered_shell_log.count())
print 'Number non-empty editor log: {}'.format(filtered_editor_log.count())

tmp_data = shell_log.map(lambda x: x.get_operation_list().count())
print 'Average length of shell operation: {}'.format(np.average(tmp_data))
tmp_data = filtered_shell_log.map(lambda x: x.get_operation_list().count())
print 'Average length of non-empty shell operation: {}'.format(np.average(tmp_data))
tmp_data = editor_log.map(lambda x: x.get_operation_list().count())
print 'Average length of editor operation: {}'.format(np.average(tmp_data))
tmp_data = filtered_editor_log.map(lambda x: x.get_operation_list().count())
print 'Average length of non-empty editor entry: {}'.format(np.average(tmp_data))

cmd_data = filtered_shell_log.map(lambda x: x.combine_shell_input())
editor_cmd_data = filtered_editor_log.map(lambda x: x.filter_editor_log(['insert', 'remove', 'paste', 'copy', 'save', 'open'])).map(lambda x: x.combine_editor_input())

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

# example_1 = example[1]
# example_2 = editor_cmd_data.find_by(lambda x: x.user_name == example_1.user_name and str(x.timestamp) == str(example_1.timestamp))
# with open('wei_shell.html', 'w') as f_out:
#     f_out.write(report_tools.print_log_item(example_1))
# if example_2:
#     with open('wei_editor.html', 'w') as f_out:
#         f_out.write(report_tools.print_log_item(example_2))

# example_2 = cmd_data.find_by(lambda x: )

# analyzer.overall_frequency(filtered_shell_log)
# analyzer.student_frequency(filtered_shell_log)
# analyzer.student_freq_clustering(filtered_shell_log, user_info)
# analyzer.student_2_means(filtered_shell_log, user_info)
# analyzer.freq_statistics(filtered_shell_log, user_info)
# analyzer.cmd_counting(cmd_data)
# analyzer.cmd_basics(cmd_data)
# analyzer.editor_basics(filtered_editor_log)
