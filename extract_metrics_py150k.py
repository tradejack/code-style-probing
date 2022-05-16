import re
import json

from tqdm.auto import tqdm
import pandas as pd

from config import PY150K_TRAIN_AST, PY150K_TRAIN_CODE, PY150K_CODE_DIR

IDENTIFIER_AST_NODE_TYPES = [
    "NameParam",
    "NameStore",
    "attr",
    "identifier",
    "keyword",
    "kwarg",
    "vararg",
    "FunctionDef",
    "ClassDef",
]


class ASTNode:
    """AST Node class specifically for loading Py150k annotated AST"""

    def __init__(self, idx, node_type, value):
        self.idx = idx
        self.type = node_type
        self.value = value
        self.children = []

    def __repr__(self):
        return f"ID:{self.idx}, {self.type}:{self.value}, children:{self.children}"


def create_ast_node(node_idx, node_dict, node_arr):
    node_type = node_dict["type"]
    node_value = node_dict.get("value", None)
    if node_value == "None":
        node_value = None
    node = ASTNode(node_idx, node_type, node_value)
    for child_node_idx in node_dict.get("children", []):
        child_node_dict = node_arr[child_node_idx]
        node.children.append(
            create_ast_node(child_node_idx, child_node_dict, node_arr)
        )
    return node


def ast_str2tree(ast_str):
    node_arr = json.loads(ast_str)
    root = create_ast_node(0, node_arr[0], node_arr)
    return root


def ast_walk(root: ASTNode):
    yield root
    if len(root.children) == 0:
        return
    for child_node in root.children:
        for node in ast_walk(child_node):
            yield node


def read_py150k_ast(filename, limit=None):
    ast = []
    count = 0
    with open(filename) as json_file:
        for line in tqdm(json_file):
            ast.append(line)
            count += 1
            if limit and count >= limit:
                break
    return ast


def read_py150k_code(filename, limit=None):
    filenames = []
    with open(filename, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            filenames += [line.strip()]
            if limit and len(filenames) >= limit:
                break

    return filenames


def read_file_to_string(filename):
    f = open(filename, "rb")
    s = ""
    try:
        s = f.read()
    except:
        print(filename)
    f.close()
    return s.decode(errors="replace")


def casing_match(token: str):
    """Identify the casing given the token

    Args:
        token (str): token of code

    Returns:
        str: the type of casing
    """
    lower_camel = r"^[a-z]+([A-Z][a-z0-9]+)+$"
    upper_camel = r"^[A-Z][a-z]+([A-Z][a-z0-9]+)*$"
    snake = r"[a-z0-9]*_*(_[a-z0-9]*)+_*"
    lower = r"^[a-z][a-z0-9]*$"
    upper = r"^[A-Z][A-Z0-9]*$"

    if re.match(snake, token):
        return "snake"
    elif re.match(upper_camel, token):
        return "upper_camel"
    elif re.match(lower_camel, token):
        return "lower_camel"
    elif re.match(lower, token):
        return "lower"
    elif re.match(upper, token):
        return "upper"
    else:
        return None


def count_identifier(node, results):
    total_result = results[0]
    var_result = results[1]
    class_result = results[2]
    func_result = results[3]
    if node.type in IDENTIFIER_AST_NODE_TYPES:
        if node.type == "ClassDef":
            result = class_result
        elif node.type == "FunctionDef":
            result = func_result
        else:
            result = var_result

        result["total"] = result.get("total", 0) + 1
        total_result["total"] = total_result.get("total", 0) + 1
        token = node.value

        if token == None:
            return

        casing = casing_match(token)

        if casing == "snake":
            result["snake"] = result.get("snake", 0) + 1
            total_result["snake"] = total_result.get("snake", 0) + 1
        elif casing == "lower":
            result["lower"] = result.get("lower", 0) + 1
            total_result["lower"] = total_result.get("lower", 0) + 1
        elif casing == "upper":
            result["upper"] = result.get("upper", 0) + 1
            total_result["upper"] = total_result.get("upper", 0) + 1
        elif casing == "lower_camel":
            result["lower_camel"] = result.get("lower_camel", 0) + 1
            total_result["lower_camel"] = (
                total_result.get("lower_camel", 0) + 1
            )
        elif casing == "upper_camel":
            result["upper_camel"] = result.get("upper_camel", 0) + 1
            total_result["upper_camel"] = (
                total_result.get("upper_camel", 0) + 1
            )
        else:
            result["none_of_these"] = result.get("none_of_these", 0) + 1
            total_result["none_of_these"] = (
                total_result.get("none_of_these", 0) + 1
            )


def get_docstring(node):
    doc_strs = []
    for body_node in node.children:
        if body_node.type != "body":
            continue
        for expr_node in body_node.children:
            if expr_node.type != "Expr":
                continue
            for str_node in expr_node.children:
                doc_str = str_node.value
                if doc_str == None:
                    continue
                doc_strs.append(doc_str)
    return doc_strs


def count_docstring(node, doc_result):
    doc_strs = get_docstring(node)
    if not doc_strs:
        return
    doc_result["count"] += len(doc_strs)
    for doc_str in doc_strs:
        # Total docstring lines
        doc_result["lines"] += len(doc_str.split("\n"))
        # Avg. docstring length in character-level
        doc_result["char_len"] += len(doc_str)
        # Avg. docstring length in word-level
        doc_result["word_len"] += len(doc_str.split())

    _id = node.type
    # # methods with docstrings
    if _id == "FunctionDef":
        doc_result["func_docstr_count"] += 1


def identifier_count_to_proportion(result):
    proportion_results = {}
    id_types = result.keys()
    for id_type in id_types:
        id_type_result = result[id_type]
        proportion_result = {}
        for case in id_type_result.keys():
            if case == "total":
                continue
            proportion_result[case] = (
                round(id_type_result[case] / id_type_result["total"], 6)
                if id_type_result["total"] > 0
                else 0
            )
        proportion_results[id_type] = proportion_result

    return proportion_results.copy()


def extract_metric_from_ast(root):
    result = {}
    id_result_template = {
        "total": 0,
        "snake": 0,
        "lower": 0,
        "upper": 0,
        "lower_camel": 0,
        "upper_camel": 0,
        "none_of_these": 0,
    }
    id_results = [
        id_result_template.copy(),
        id_result_template.copy(),
        id_result_template.copy(),
        id_result_template.copy(),
    ]
    doc_result = {
        "lines": 0,
        "char_len": 0,
        "word_len": 0,
        "count": 0,
        "func_docstr_count": 0,
    }
    for node in ast_walk(root):
        # Count Identifiers
        count_identifier(node, id_results)
        # Count Docstring
        count_docstring(node, doc_result)

    # Avg. docstring length
    doc_result["char_len"] = (
        round(doc_result["char_len"] / doc_result["count"], 6)
        if doc_result["count"] > 0
        else 0
    )
    doc_result["word_len"] = (
        round(doc_result["word_len"] / doc_result["count"], 6)
        if doc_result["count"] > 0
        else 0
    )

    id_result = identifier_count_to_proportion(
        {
            "total": id_results[0],
            "var": id_results[1],
            "class": id_results[2],
            "func": id_results[3],
        }
    )
    result["id"] = id_result.copy()
    result["doc_str"] = doc_result.copy()

    return result.copy()


def count_comments(input_code):
    comment = r"#.*"
    search = re.findall(comment, input_code)
    comment_len = 0
    for comment in search:
        comment_len += len(comment)
    return len(search), comment_len


def create_df(data):
    data_df = pd.DataFrame(data, columns=["file_name"])

    # get the repo and username from script file name
    script_file_name_regex = re.compile(r"data/([^/]+)/([^/]+)/.+")

    usernames = []
    repos = []
    py_scripts = []
    comments = []
    comment_lens = []
    comment_dens = []
    line_counts = []

    for file_name in tqdm(data_df["file_name"]):

        match = script_file_name_regex.search(file_name)
        if not match:
            print(file_name)

        username = match.group(1)
        repo_name = match.group(2)

        file_string = read_file_to_string(f"{PY150K_CODE_DIR}/{file_name}")
        line_count = len(file_string.split("\n"))

        comment_count, comment_len = count_comments(file_string)
        comment_den = (
            round(comment_count / line_count, 6) if line_count > 0 else 0
        )

        usernames += [username]
        repos += [repo_name]
        py_scripts += [file_string]
        comments += [comment_count]
        comment_lens += [comment_len]
        comment_dens += [comment_den]
        line_counts += [line_count]

    data_df["user_name"] = usernames
    data_df["repo_name"] = repos
    data_df["py_script"] = py_scripts
    data_df["comment"] = comments
    data_df["comment_len"] = comment_lens
    data_df["comment_den"] = comment_dens
    data_df["line_count"] = line_counts

    return data_df


if __name__ == "__main__":
    # sample code for loading Py150k ast strings, remove limit for loading all data
    sample_ast_str_list = read_py150k_ast(PY150K_TRAIN_AST, limit=20)
    print(sample_ast_str_list[18])

    # sample code for extracting metrics from Py150K AST
    data_metrics = []
    for sample_ast_str in sample_ast_str_list:
        ast_tree = ast_str2tree(sample_ast_str)
        data_metric = extract_metric_from_ast(ast_tree)
        data_metrics.append(data_metric)
    print(data_metrics[18])

    # sample code for loading Py150k code
    sample_code_filenames = read_py150k_code(PY150K_TRAIN_CODE, limit=20)
    print(sample_code_filenames[18])

    # sample code for extracting metrics from Py150K script
    sample_data_df = create_df(sample_code_filenames)
    print(sample_data_df[18])
