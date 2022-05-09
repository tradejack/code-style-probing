import json

from tqdm.auto import tqdm


class ASTNode:
    def __init__(self, node_type, value):
        self.type = node_type
        self.value = value
        self.children = []

    def __repr__(self):
        return f"{self.type}:{self.value} children:{self.children}"


def load_train_data():
    """Load the training data of Py150k

    Returns:
        List: the ast string list, the string can be parsed to JSON/dict
    """
    ast = []
    with open("../data/py150/python100k_train.json") as json_file:
        for line in tqdm(json_file):
            ast.append(line)

    return ast


def create_ast_node(node_dict, node_arr):
    """Create the AST node given nodes array and the current node dict

    Args:
        node_dict (Dict): node dict
        node_arr (List): array of all node dict

    Returns:
        ASTNode: the constructed node object for AST
    """
    node_type = node_dict["type"]
    node_value = node_dict.get("value", None)
    node = ASTNode(node_type, node_value)
    for child_node_idx in node_dict.get("children", []):
        child_node_dict = node_arr[child_node_idx]
        node.children.append(create_ast_node(child_node_dict, node_arr))
    return node


def ast_str2tree(ast_str):
    """convert AST string array to AST Tree, where the node class is defined above

    Args:
        ast_str (str): the full AST string provided by Py150k dataset, will be converted to a list of node dict

    Returns:
        ASTNode: the root node of the AST
    """
    node_arr = json.loads(ast_str)
    root = create_ast_node(node_arr[0], node_arr)
    return root


def ast_walk(root):
    """A generator function for walking through AST

    Args:
        root (ASTNode): the root of AST

    Yields:
        ASTNode: every node in AST
    """
    yield root
    if len(root.children) == 0:
        return
    for child_node in root.children:
        for node in ast_walk(child_node):
            yield node