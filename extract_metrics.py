"""
Extract Metric script for Python Code
To make the extracted metric consistent, put the code here
"""
import ast
from collections import Counter

from utils.helper import read_file_to_string, calculate_ratio
from utils.regex_parse import casing, comment, CASING_REGEX_MAPPING
from utils.ast_parse import ast_parse

FUNC_TYPES = [ast.AsyncFunctionDef, ast.FunctionDef]

# Documentation
# Comments
def count_comment(code, counter):
    comment_matches = comment(code)
    counter["comment_count"] += len(comment_matches)
    for comment_match in comment_matches:
        counter["comment_total_len"] += len(comment_match)
    counter["comment_avg_len"] += calculate_ratio(
        counter["comment_total_len"], counter["comment_count"]
    )
    counter["comment_density"] += calculate_ratio(
        counter["comment_count"], counter["line_count"]
    )


# Docstrings
def count_docstring(ast_node, counter):
    # if we find a function or class name, check for docstrings
    if type(ast_node) not in ([ast.ClassDef] + FUNC_TYPES):
        return
    doc = ast.get_docstring(ast_node)
    if not doc:
        return
    counter["ds_count"] += 1
    counter["ds_char_len_total"] += len(doc)
    counter["ds_word_len_total"] += len(doc.split())
    counter["ds_line_count"] += len(doc.split("\n"))
    if type(ast_node) == ast.FunctionDef:
        counter["ds_of_method"] += 1


def count_docstring_density(counter):
    counter["ds_density"] = calculate_ratio(
        counter["ds_line_count"], counter["line_count"]
    )


def count_docstring_average(counter):
    counter["ds_char_len_avg"] = calculate_ratio(
        counter["ds_char_len_total"], counter["ds_count"]
    )
    counter["ds_word_len_avg"] = calculate_ratio(
        counter["ds_word_len_total"], counter["ds_count"]
    )


# Formatting
# Casing Counts
def count_casing(ast_node, counter):
    # if we find a variable, count its casing
    if type(ast_node) != ast.Name:
        return
    counter["id_total"] += 1
    case = casing(ast_node.id)
    if case:
        counter[case] += 1


def count_casing_ratio(counter):
    for case in CASING_REGEX_MAPPING.keys():
        counter[f"{case}_ratio"] = calculate_ratio(
            counter[case], counter["id_total"]
        )

    counter["other_case_ratio"] = calculate_ratio(
        counter["other_case"], counter["id_total"]
    )


# Line Metrics
def count_lines(code, counter):
    counter["line_count"] += len(code.split("\n"))


# Methods/Classes
# Method Metrics
def count_method(ast_node, counter):
    # if func def found see how many parents and decorators
    if type(ast_node) not in FUNC_TYPES:
        return
    counter["func_count"] += 1
    counter["func_decorators_count"] += len(ast_node.decorator_list)
    if type(ast_node) != ast.AsyncFunctionDef:
        return
    counter["func_async_count"] += 1


def count_method_ratio(counter):
    counter["func_decotrators_ratio"] = calculate_ratio(
        counter["func_decorators_count"], counter["func_count"]
    )
    counter["func_async_ratio"] = calculate_ratio(
        counter["func_async_count"], counter["func_count"]
    )


# Class Metrics
def count_class(ast_node, counter):
    # if class def found see how many parents and decorators
    if type(ast_node) != ast.ClassDef:
        return
    counter["class_count"] += 1
    counter["class_parents_count"] += len(ast_node.bases)
    counter["class_decorators_count"] += len(ast_node.decorator_list)


def count_class_ratio(counter):
    counter["class_parents_ratio"] = calculate_ratio(
        counter["class_parents_count"], counter["class_count"]
    )
    counter["class_decorators_ratio"] = calculate_ratio(
        counter["class_decorators_count"], counter["class_count"]
    )


# Python Language Features


def extract_metrics(code):

    metrics = Counter()
    ast_tree = ast_parse(code)

    extract_funcs_from_code = [count_lines, count_comment]
    extract_funcs_from_ast = [
        count_casing,
        count_docstring,
        count_class,
        count_method,
    ]
    extract_funcs_from_counter = [
        count_casing_ratio,
        count_method_ratio,
        count_class_ratio,
        count_docstring_density,
        count_docstring_average,
    ]

    for extract_func in extract_funcs_from_code:
        extract_func(code, metrics)

    for node in ast.walk(ast_tree):
        for extract_func in extract_funcs_from_ast:
            extract_func(node, metrics)

    for extract_func in extract_funcs_from_counter:
        extract_func(metrics)

    return metrics


if __name__ == "__main__":
    # Python3 test code
    test_code = read_file_to_string("./utils/test_code.py")
    print(extract_metrics(test_code))
