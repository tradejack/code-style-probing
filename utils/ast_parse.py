import ast
import json


class Py150kAST:
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
        node = Py150kAST.ASTNode(node_idx, node_type, node_value)
        for child_node_idx in node_dict.get("children", []):
            child_node_dict = node_arr[child_node_idx]
            node.children.append(
                Py150kAST.create_ast_node(
                    child_node_idx, child_node_dict, node_arr
                )
            )
        return node

    def ast_str_to_tree(ast_str):
        node_arr = json.loads(ast_str)
        root = Py150kAST.create_ast_node(0, node_arr[0], node_arr)
        return root

    def ast_walk(root):
        yield root
        if len(root.children) == 0:
            return
        for child_node in root.children:
            for node in Py150kAST.ast_walk(child_node):
                yield node


def get_ast_node_type(node, py150k=False):
    if py150k:
        return node.type
    return type(node)


def get_func_type(py150k=False):
    if py150k:
        return ["FunctionDef"]
    return [ast.FunctionDef, ast.AsyncFunctionDef]


def get_async_func_type(py150k=False):
    if py150k:
        # There is no async function for Python2
        return []
    return [ast.AsyncFunctionDef]


def get_class_type(py150k=False):
    if py150k:
        return ["ClassDef"]
    return [ast.ClassDef]


def get_var_type(py150k=False):
    if py150k:
        return [
            "NameParam",
            "NameStore",
            "attr",
            "identifier",
            "keyword",
            "kwarg",
            "vararg",
        ]
    return [ast.Name]

def get_comp_type(py150k=False):
    if py150k:
        return ["ListComp", "DictComp", "SetComp"]
    return [ast.ListComp, ast.DictComp, ast.SetComp]

def get_generator_type(py150k=False):
    if py150k:
        return ['GeneratorExp']
    return [ast.GeneratorExp]

def get_lambda_type(py150k=False):
    if py150k:
        return ["Lambda"]
    return [ast.Lambda]

def get_docstring(node, py150k=False):
    if py150k:
        doc_strs = ""
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
                    doc_strs += doc_str + "\n"
        return doc_strs
    return ast.get_docstring(node)


def get_parents(node, py150k=False):
    if py150k:
        bases = []
        for child_node in node.children:
            if child_node.type == "bases":
                bases += child_node.children
        return bases
    return node.bases


def get_decorators(node, py150k=False):
    if py150k:
        decorator_list = []
        for child_node in node.children:
            if child_node.type == "decorator_list":
                decorator_list += child_node.children
        return decorator_list
    return node.decorator_list
