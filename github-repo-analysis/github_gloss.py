# -*- coding: utf-8 -*-
"""from github_gloss.ipynb

Original file is located at
    https://colab.research.google.com/drive/1AbskOFQkoWuXOFAf1nIS6vEdalpZ84Mv
"""
from cgi import test
import matplotlib.pyplot as plt
from pydoc import doc
import re 
import os
from collections import Counter
import pandas as pd
from ast_gloss import ast_parse
from tqdm import tqdm
import ast

test_code = "def function_name(par_1, parTwo, camelCase):\n\t\"\"\"\n\tdocstring time\n\t\"\"\"\n\tvar_1 = 42 # cool and awesome comment\n\tprint('hello world!') #comment too\n\treturn "
# print(test_code)

# Use AST package for better parsing! 

def count_casing(input_code):
    case_dict = {"lowerCamelCase": 0, "UpperCamelCase": 0, "snake_case": 0}
    lower_camel = r"[a-z]+([A-Z][a-z0-9]+)+"
    upper_camel = r"[A-Z][a-z]+([A-Z][a-z0-9]+)+"
    snake = r"[a-z]+(_[a-z0-9]+)+"
    input_code = input_code.replace("(", " ")
    input_code = input_code.replace(")", " ")
    for token in input_code.split():
        if re.match(lower_camel, token):
            case_dict["lowerCamelCase"] += 1
        elif re.match(upper_camel, token):
            case_dict["UpperCamelCase"] += 1
        elif re.match(snake, token):
            case_dict["snake_case"] += 1
    return case_dict


def count_docstrings(input_code):
    docstr = r"\"\"\"[\s\S]*?\"\"\"|\'\'\'[\s\S]*?\'\'\'"
    search = re.findall(docstr, input_code)
    doc_len = 0
    for docstring in search:
        doc_len += len(docstring)
    return len(search), doc_len


def count_comments(input_code):
    comment = r"#.*"
    search = re.findall(comment, input_code)
    comment_len = 0
    for comment in search:
        comment_len += len(comment)
    return len(search), comment_len

def count_whitespace(input_code):
  non_space_lines = len(input_code.split())
  spaces = 0
  for char in input_code:
    if char == "\n":
      spaces += 1
  return (spaces-non_space_lines)/spaces

def count_type_comments(input_code):
  parse = ast.parse(input_code, type_comments = True)
  ast.dump(parse)
  comments = 0
  for node in ast.walk(parse):
    id = type(node) 
    if id == ast.PyCF_TYPE_COMMENTS: # if we find a variable, count its casing
      comments += 1
  return comments

def count_lines_per_method(input_code):
    parse = ast.parse(input_code, type_comments = True)
    ast.dump(parse)
    print(parse)
    comments = 0
    for node in ast.walk(parse):
      id = type(node) 
      if id == ast.PyCF_TYPE_COMMENTS: # if we find a variable, count its casing
        comments += 1
    return comments
    
print(count_lines_per_method(test_code))
# test the regex searching methods:
# print(f"Casing  count: {count_casing(test_code)}")
# print(f"Docstr  count: {count_docstrings(test_code)}")
# print(f"Comment count: {count_comments(test_code)}")

def repo_probe(directory):
  """
  function that takes a repo folder, walks through the FS, and calculates
  the following stats:
    * # of instances of casing
    * # of docstrings TODO - Count docstrings per method / how many methods have docstrings
    * # of comments
    * Average docstring length (words)
    * Average comment length (words)
    * Average comment density
  """
  comment_count = 0
  comment_length = 0
  docstring_count = 0
  docstring_length = 0
  line_count = 0
  comment_density =  0
  docstring_density =  0

  eval_dict = Counter()

  for root, _, files in os.walk(directory):
      for file_name in files:
          if file_name.endswith(".py"):
            file_path = os.path.join(root, file_name)
            with(open(file_path, 'r', encoding='latin-1')) as fp:
              code = fp.read()
              fp.close()

              line_count += sum([1 for _ in code.split("\n")])

              # calculate stats
              num_comments, comments_lengths = count_comments(code)

              if num_comments: # if there were any comments in this file; update
                comment_count  += num_comments
                comment_length += comments_lengths/num_comments
              
              eval_dict['total_num_files'] += 1
              try:
                parse = ast_parse(code)
                eval_dict += parse
              except:
                print(f"Parse error on file {file_path}, skipping.") #TODO try different encodings to get special characters

  #TODO: Normalize new AST counts over number of files/lines etc to get better stuff. make more graphs
  file_count = eval_dict["total_num_files"]
  var_count = eval_dict["var_total"]
  eval_dict['average_comment_count']  =  comment_count    / file_count  # average comment count over files        
  eval_dict['average_comment_length'] =  comment_length   / comment_count # average length of comments
  eval_dict['comment_line_density']   =  comment_count / line_count # average number of commented lines

  total_methods = eval_dict[ast.FunctionDef] + eval_dict[ast.ClassDef]
  if total_methods: # normalizes docstrings over number of classes AND functions 
    eval_dict['methods_with_docstrings'] = eval_dict["ds_count"] / total_methods

  if var_count: # gets casing ratio over all variable counts
    eval_dict['ratio_snake_case'] = eval_dict['snake_case'] / var_count
    eval_dict['ratio_locamel_case'] = eval_dict['lower_camel'] / var_count
    eval_dict['ratio_upcamel_case'] = eval_dict['upper_camel'] / var_count
    eval_dict['ratio_lower_case'] = eval_dict['lower'] / var_count

  eval_dict['lines_per_file']   =  line_count / eval_dict["total_num_files"]

  return eval_dict

repo_evals = {}

personal = ["npy_datetime", "shopi", "Gnome-menu-applet", "Calculator-Course-2019", "vivo-remove-people", "cjk-defn", "ir-reduce", "fret_benchmark", "googlepersonfinder", "python_chess"]
professional =  ['awesome-python', 'django', 'flask', 'keras', 'nltk', 'pandas', 'pytorch', 'scikit-learn', 'scipy', 'youtube-dl']
all_repos =   personal + professional
all_repos_normal = [repo for repo in all_repos if repo not in ['awesome-python', 'python_chess'] ] 
if False:
  for repo in all_repos_normal:
    print(f"Scanning repo {repo}...")
    path = "data/"+repo
    repo_evals[repo] = repo_probe(path)

  eval_df = pd.DataFrame(repo_evals)
  print(eval_df)

# TODO: prune out no-docstring documents

def graph_stats(x , y, title, x_axis, y_axis, path):
  """
  Plots 
  """
  fig = plt.figure()
  ax = fig.add_axes([0,0,1,1])
  ax.bar(x, y, color=['r']*(len(x)//2)+['b']*(len(x)//2) )
  plt.title(title)
  plt.ylabel(y_axis)
  plt.xlabel(x_axis)
  plt.xticks(rotation=90)
  plt.savefig(path+".png", bbox_inches='tight', pad_inches=0)

if False:
  for stat in repo_evals["npy_datetime"]:
    if stat != "casing_count":
      title = " ".join(stat.split('_')).title()
      graph_stats(all_repos_normal, eval_df.loc[stat,], title +" across Repos", "Repositories", title, "graphs/"+stat)

  snake_case = [lookup['snake_case'] for lookup in eval_df.loc['casing_count']]
  lower_camel_case = [lookup['lowerCamelCase'] for lookup in eval_df.loc['casing_count']]
  upper_camel_case = [lookup['UpperCamelCase'] for lookup in eval_df.loc['casing_count']]

  graph_stats(all_repos_normal, snake_case, "Snake Case count" +" across Repos", "Repositories", "Snake Case count", "graphs/"+"Snake-Case-count")
  graph_stats(all_repos_normal, lower_camel_case, "lower Camel Case count" +" across Repos", "Repositories", "lower Camel Case count", "graphs/"+"lower-Case-count")
  graph_stats(all_repos_normal, upper_camel_case, "Upper Camel Case count" +" across Repos", "Repositories", "Upper Camel Case count", "graphs/"+"upper-Case-count")