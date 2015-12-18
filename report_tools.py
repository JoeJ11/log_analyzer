import json
import os
import matplotlib.pyplot as plt
import data_reader
# import numpy as np
# from scipy import stats

def prepare_plot(figsize=(8.5, 4), hideLabels=False, gridColor='#999999', gridWidth=1.0):
    '''
        Template to generate plot figure
    '''
    plt.close()
    fig, ax = plt.subplots(figsize=figsize, facecolor='white', edgecolor='white')
    ax.set_frame_on(True)
    # ax.lines[0].set_visible(False)
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    # ax.get_yaxis().set_visible(False)
    ax.axes.tick_params(labelcolor='black', labelsize='10')
    plt.grid(color=gridColor, linewidth=gridWidth, linestyle='-')
    # plt.gcf().subplots_adjust(bottom=0.5)
    # map(lambda position: ax.spines[position].set_visible(False), ['bottom', 'top', 'left', 'right'])
    return fig, ax

def print_log_item(log_item):
    str_result = ''
    if log_item.log_type == data_reader.LOG_TYPE_SHELL:
        for index, item in enumerate(log_item.cmd_list):
            str_result += "<h3 style=\"color:blue\">{}</h3>\n".format(item)
            if log_item.has_timestamp:
                str_result += "<p>timestamp: {}-{}</p>\n".format(log_item.timestamp_list[index][0], log_item.timestamp_list[index][1])
    elif log_item.log_type == data_reader.LOG_TYPE_EDITOR:
        for item in log_item.cmd_list:
            if item['action'] == 'insert':
                str_result += u"<h3 style=\"color:green\">insert</h3>\n"
            elif item['action'] == 'remove':
                str_result += u"<h3 style=\"color:red\">remove</h3>\n"
            elif item['action'] == 'copy':
                str_result += u"<h3 style=\"color:yellow\">copy</h3>\n"
            elif item['action'] == 'paste':
                str_result += u"<h3 style=\"color:gray\">paste</h3>\n"
            elif item['action'] in ['open', 'save']:
                str_result += u"<h3 style=\"color:purple\">{}</h3>\n".format(item['action'])
            else:
                continue
            if item['action'] in ['insert', 'remove']:
                str_result += u"<p>{}</p>\n".format(item['lines'])
            elif item['action'] in ['paste', 'copy']:
                str_result += u"<p>{}</p>\n".format(item['text'])
    return str_result

def print_log_item_with_time(log_item):
    result = []
    if log_item.log_type == data_reader.LOG_TYPE_SHELL:
        for index, item in enumerate(log_item.cmd_list):
            str_result = ''
            str_result += "<h3 style=\"color:blue\">{}</h3>\n".format(item)
            result.append((str_result, int(log_item.timestamp_list[index][1])))
    elif log_item.log_type == data_reader.LOG_TYPE_EDITOR:
        for item in log_item.cmd_list:
            str_result = ''
            if item['action'] == 'insert':
                str_result += u"<h3 style=\"color:green\">insert</h3>\n"
            elif item['action'] == 'remove':
                str_result += u"<h3 style=\"color:red\">remove</h3>\n"
            elif item['action'] == 'copy':
                str_result += u"<h3 style=\"color:yellow\">copy</h3>\n"
            elif item['action'] == 'paste':
                str_result += u"<h3 style=\"color:gray\">paste</h3>\n"
            else:
                continue
            if item['action'] in ['insert', 'remove']:
                str_result += u"<p>{}</p>\n".format(item['lines'])
            else:
                str_result += u"<p>{}</p>\n".format(item['text'])
            result.append((str_result, int(item['timestamp']/1000)))
    return result
