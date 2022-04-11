#imports
import math
from github import Github
from os import walk
from os import listdir
import os
import glob
import re

PATH = './repos'

#functions
def casing_count(file_lines):
    counts = 0
    case_dict = {"camelCase":0, "snake_case":0}
    camel = r"[a-z]+([A-Z][a-z0-9]+)+"
    snake = r"[a-z]+(_[a-z0-9]+)+"
    file_lines = ' '.join(file_lines)
    file_lines = file_lines.replace("("," ")
    file_lines = file_lines.replace(")"," ")
    #print (file_lines)

    for token in file_lines.split():
        if re.match(camel, token):
            case_dict["camelCase"] += 1
        elif re.match(snake, token):
          case_dict["snake_case"] += 1
    #print (case_dict)
    #9 / 0
    return case_dict
    
def comment_analysis(file_lines):
    #maybe we modify this so that it can pick up comments at end of lines all though this is rare in high quality code
    #use a search and then return index to show where comment is starting
    comment_count = 0
    comment_len = 0
    for line in file_lines:
        line = line.lstrip()
        if line != '':
            if line[0] == '#':
                comment_count += 1
                comment_len += len(line)
    #if comment_count != 0:
    #    comment_len = comment_len / comment_count
    #print (comment_count, comment_avg)
    eval_dict = {'comment_count' : comment_count, 'comment_len' : comment_len}
    return eval_dict
    
def docstring_analysis(file_lines):
    input_code = ' '.join(file_lines)
    #print (type(input_code))
    docstr = r"\"\"\"[\s\S]*?\"\"\""
    search = re.findall(docstr, input_code)
    eval_dict = {'doc_count':0, 'doc_len':0}
    for doc in search:
        eval_dict['doc_count'] +=1
        eval_dict['doc_len'] +=len(doc)
    #if eval_dict['doc_count'] != 0:
    #    eval_dict['doc_len'] = eval_dict['doc_len'] / eval_dict['doc_count']
    return eval_dict
    
def file_analysis(file_name): 
    #print (file_name)
    file_count = 0
    try:
        with open(file_name) as f:
            lines = f.readlines()
            file_count +=1
        #print (lines)
    except:
        #print ("error opening" + file_name)
        return -1
    case = casing_count(lines)
    com = comment_analysis(lines)
    doc = docstring_analysis(lines)

    return (case, com, doc)
    
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]  
            
def repo_analysis(repo_name, path):
    #cycle through 
    updated_stats = {'files': 0, "camelCase":0, "snake_case":0, 'comment_count' : 0, 'comment_len' : 0, 'doc_count' : 0, 'doc_len' : 0}
    sub_dirs = get_immediate_subdirectories(path + repo_name)
    py_files = (glob.glob(path + repo_name + "\*.py"))
    
    for file in py_files:
        file = file.replace('\\' , '/')
        analysis_out = file_analysis(file)
        if analysis_out != -1:
            updated_stats['files'] +=1
            updated_stats["camelCase"] += analysis_out[0]["camelCase"]
            updated_stats["snake_case"] += analysis_out[0]["snake_case"]
            updated_stats['comment_count'] += analysis_out[1]['comment_count']
            updated_stats['comment_len'] += analysis_out[1]['comment_len']
            #print (analysis_out[0])
            updated_stats['doc_count'] += analysis_out[2]['doc_count']
            updated_stats['doc_len'] += analysis_out[2]['doc_len'] # fix

    
    #recursively call
    
    for directory in sub_dirs:
        child_stats = repo_analysis(directory, (path + repo_name + '/'))
        updated_stats['files'] += child_stats['files']
        updated_stats["camelCase"] += child_stats["camelCase"]
        updated_stats["snake_case"] += child_stats["snake_case"]
        updated_stats['comment_count'] += child_stats['comment_count']
        updated_stats['comment_len'] += child_stats['comment_len']
        #print (analysis_out[0])
        updated_stats['doc_count'] += child_stats['doc_count']
        updated_stats['doc_len'] += child_stats['doc_len'] # fix
    
    #if len(sub_dirs) != 0: 
    #    updated_stats['doc_len'] = updated_stats['doc_len'] / len(sub_dirs)
    #    updated_stats['comment_len'] = updated_stats['comment_len'] /  len(sub_dirs)
        
    #handle python files

    return updated_stats
    
def graph_stats(red_x_data, red_y_data, blue_x_data, blue_y_data, title, x_axis, y_axis):
  """
  Plots 
  """
  fig = plt.figure()
  ax = fig.add_axes([0,0,1,1])
  ax.bar(red_x_data+blue_x_data, red_y_data + blue_y_data, color=['r']*len(red_x_data)+['b']*len(blue_x_data))
  plt.title(title)
  plt.ylabel(y_axis)
  plt.xlabel(x_axis)
  plt.xticks(rotation=90)
  plt.show()

repos = listdir(PATH)
print (repos)
for repo in repos:
    updated_stats = repo_analysis(repo, PATH + '/')

    if updated_stats['doc_count'] != 0: 
        updated_stats['doc_len'] = updated_stats['doc_len'] / updated_stats['doc_count']
    if updated_stats['comment_count'] != 0: 
        updated_stats['comment_len'] = updated_stats['comment_len'] /  updated_stats['comment_count']
    updated_stats['doc_count'] = updated_stats['doc_count'] / updated_stats['files']
    updated_stats['comment_count'] = updated_stats['comment_count'] / updated_stats['files']
    print (repo)
    print ((updated_stats))