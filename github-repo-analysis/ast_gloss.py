import ast
import matplotlib.pyplot as plt
from pydoc import doc
import re 
import os
from collections import Counter
import pandas as pd

test_code = "def function_name(par_1, parTwo, camelCase):\n\t\"\"\"\n\tdocstring time\n\t\"\"\"\n\tvar_1 = 42 # cool and awesome comment\n\tprint('hello world!') #comment too\n\treturn "

def ast_parse(path):
    """
    Tracks:
        Docstring raw count, char length, word length, function density
        Variable  raw count, char length, casing, word length (if possible)
    """
    parse = ast.parse(path)
    ast.dump(parse)
    targets = [ast.Name , ast.FunctionDef, ast.ClassDef]
    id_counter = Counter()
    for node in ast.walk(parse):
        id = type(node) 
        if id in targets: 
            if id == ast.Name: # if we find a variable, count its casing
                id_counter['var_total'] += 1
                case = casing(node.id)
                if case:
                    id_counter[case] += 1
            else: # if we find a function or class name, check for docstrings
                doc = ast.get_docstring(node)
                if doc:
                    id_counter["ds_count"] += 1
                    id_counter["ds_char_len"] += len(doc)
                    id_counter["ds_word_len"] += len(doc.split())
            id_counter[type(node)] += 1
    return id_counter

test_text = ['snake_case', 'lowerCamelCase', 'UpperCamelCase', 'lower', 'none-of-these']

def casing(token):
    """
    returns the casing of the input text, as well as any subword splits
    Input: string
    Output: string, denoting
        snake_case, lowerCamelCase, upperCamelCase, lower, else None
    """
    lower_camel = r"^[a-z]+([A-Z][a-z0-9]+)+$"
    upper_camel = r"^[A-Z][a-z]+([A-Z][a-z0-9]+)+$"
    snake = r"[a-z]+(_[a-z0-9]+)+"
    if re.match(snake, token):
        return "snake_case"
    elif re.match(upper_camel, token):
        return "upper_camel"
    elif re.match(lower_camel, token):
        return "lower_camel"
    elif re.match( r"^[a-z]+$", token):
        return "lower"
    else:
        return None

#print(ast_parse(test_code))
#for test in test_text:
    #print(casing(test))