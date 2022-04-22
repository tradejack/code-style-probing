"""
Extract Metric script for Python3 code
To make the extracted metric consistent, put the code here
"""
import ast


def count_method_with_docstring(ast_tree):
    # methods with docstrings
    count = 0
    for node in ast.walk(ast_tree):
        _id = type(node)
        if _id == ast.FunctionDef:
            docstr = ast.get_docstring(node)
            if docstr:
                count += 1
    return count