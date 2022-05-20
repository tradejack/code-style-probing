"""
Extract Metric script for Python Code
To make the extracted metric consistent, put the code here
"""
import ast
import sys
from collections import Counter

from tqdm.auto import tqdm

from utils.helper import (
    read_file_to_string,
    calculate_ratio,
    read_py150k_ast,
    read_py150k_code,
    metric_dict_to_df,
)
from utils.regex_parse import casing, comment, CASING_REGEX_MAPPING
from utils.ast_parse import (
    Py150kAST,
    get_ast_node_type,
    get_func_type,
    get_class_type,
    get_var_type,
    get_async_func_type,
    get_docstring,
    get_parents,
    get_decorators,
    get_comp_type,
    get_lambda_type
)
from config import *


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
def count_docstring(ast_node, node_type, counter, py150k=False):
    # if we find a function or class name, check for docstrings
    if node_type not in (get_class_type(py150k) + get_func_type(py150k)):
        return
    doc = get_docstring(ast_node, py150k)
    if not doc:
        return
    counter["ds_count"] += 1
    counter["ds_char_len_total"] += len(doc)
    counter["ds_word_len_total"] += len(doc.split())
    counter["ds_line_count"] += len(doc.split("\n"))
    if node_type in get_func_type(py150k):
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
def count_casing(ast_node, node_type, counter, py150k=False):
    # if we find a variable, count its casing
    if node_type not in (
        get_var_type(py150k) + get_class_type(py150k) + get_func_type(py150k)
    ):
        return

    if py150k:
        token = ast_node.value
    elif node_type in get_var_type(py150k):
        token = ast_node.id
    else:
        token = ast_node.name

    if not token:
        return
    case = casing(token)

    counter["id_total"] += 1
    counter[case] += 1

    if node_type in get_var_type(py150k):
        counter["id_total_var"] += 1
        counter[f"{case}_var"] += 1
    if node_type in get_class_type(py150k):
        counter["id_total_class"] += 1
        counter[f"{case}_class"] += 1
    if node_type in get_func_type(py150k):
        counter["id_total_method"] += 1
        counter[f"{case}_method"] += 1


def count_casing_ratio(counter):
    for case in list(CASING_REGEX_MAPPING.keys()) + ["other_case"]:
        counter[f"{case}_ratio"] = calculate_ratio(
            counter[case], counter["id_total"]
        )
        for id_type in ["var", "class", "method"]:
            counter[f"{case}_{id_type}_ratio"] = calculate_ratio(
                counter[f"{case}_{id_type}"], counter[f"id_total_{id_type}"]
            )


# Line Metrics
def count_lines(code, counter):
    counter["line_count"] += len(code.split("\n"))


# Methods/Classes
# Method Metrics
def count_method(ast_node, node_type, counter, py150k=False):
    # if func def found see how many parents and decorators
    if node_type not in get_func_type(py150k):
        return
    counter["func_count"] += 1
    counter["func_decorators_count"] += len(get_decorators(ast_node, py150k))


def count_async_method(ast_node, node_type, counter, py150k=False):
    if node_type not in get_async_func_type(py150k):
        return
    counter["func_async_count"] += 1


def count_method_ratio(counter):
    counter["func_decorators_avg"] = calculate_ratio(
        counter["func_decorators_count"], counter["func_count"]
    )
    counter["func_async_ratio"] = calculate_ratio(
        counter["func_async_count"], counter["func_count"]
    )


# Class Metrics
def count_class(ast_node, node_type, counter, py150k=False):
    # if class def found see how many parents and decorators
    if node_type not in get_class_type(py150k):
        return
    counter["class_count"] += 1
    counter["class_parents_count"] += len(get_parents(ast_node, py150k))
    counter["class_decorators_count"] += len(get_decorators(ast_node, py150k))


def count_class_ratio(counter):
    counter["class_parents_ratio"] = calculate_ratio(
        counter["class_parents_count"], counter["class_count"]
    )
    counter["class_decorators_avg"] = calculate_ratio(
        counter["class_decorators_count"], counter["class_count"]
    )


# Python Language Features
def count_comp(ast_node, node_type, counter, py150k=False):
    if node_type not in get_comp_type(py150k):
        return
    counter["comprehensions"] += 1


# Main extraction method
def extract_metrics(code, ast_tree, py150k=False):

    metrics = Counter()

    extract_funcs_from_code = [count_lines, count_comment]
    extract_funcs_from_ast = [
        count_casing,
        count_docstring,
        count_class,
        count_method,
        count_async_method,
        count_comp,
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

    ast_walk_func = Py150kAST.ast_walk if py150k else ast.walk

    for node in ast_walk_func(ast_tree):
        node_type = get_ast_node_type(node, py150k)
        for extract_func in extract_funcs_from_ast:
            extract_func(node, node_type, metrics, py150k)

    for extract_func in extract_funcs_from_counter:
        extract_func(metrics)

    return metrics


if __name__ == "__main__":
    # Python3 test code
    print("\nTesting Python3 Code Metric Extraction\n")
    test_code = read_file_to_string("./utils/test_code.py")
    test_ast_tree = ast.parse(test_code)
    test_python3_metrics = extract_metrics(test_code, test_ast_tree)
    print(test_python3_metrics)
    print(metric_dict_to_df([test_python3_metrics]))

    # Py150k test code
    print("\nTesting Py150K Code Metric Extraction\n")
    sample_idx = 45
    sample_ast_str_list = read_py150k_ast(PY150K_TRAIN_AST, limit=60)
    sample_code_filenames = read_py150k_code(PY150K_TRAIN_CODE, limit=60)
    test_code = read_file_to_string(
        f"{PY150K_CODE_DIR}/{sample_code_filenames[sample_idx]}"
    )
    test_ast_tree = Py150kAST.ast_str_to_tree(sample_ast_str_list[sample_idx])
    test_py150k_metrics = extract_metrics(
        test_code, test_ast_tree, py150k=True
    )
    print(test_py150k_metrics)
    print(metric_dict_to_df([test_py150k_metrics]))

    if len(sys.argv) > 1 and sys.argv[1] == "py150k":
        print("\nExtracting Py150K Code Data Metrics\n")
        # Py150k all code metric extraction
        ast_str_list = read_py150k_ast(PY150K_TRAIN_AST)
        code_filenames = read_py150k_code(PY150K_TRAIN_CODE)

        py150k_metrics_list = []
        for idx in tqdm(range(len(ast_str_list))):
            code_str = read_file_to_string(
                f"{PY150K_CODE_DIR}/{code_filenames[idx]}"
            )
            ast_tree = Py150kAST.ast_str_to_tree(ast_str_list[idx])
            py150k_metrics = extract_metrics(code_str, ast_tree, py150k=True)
            py150k_metrics_list.append(py150k_metrics)

        output_df = metric_dict_to_df(py150k_metrics_list)
        output_df.to_csv("py150k_metrics.csv", index=False)
