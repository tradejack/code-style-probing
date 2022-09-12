import pandas as pd

from tqdm.auto import tqdm


import ast
import astunparse

# Comment
def uncomment(source):
    """ 
    Takes input code and returns code with comments stripped
    Input: code (str)
    Output: code (str)
    """
    try:
        parse = astunparse.unparse(ast.parse(source))
    except:
        parse = None
    return parse


# Class
def is_init_func(func_node):
    return func_node.name == "__init__"


def is_super_expr(expr_node):
    for node in ast.walk(expr_node):
        if hasattr(node, "func") and type(node.func) == ast.Name:
            if node.func.id == "super":
                return True
    return False


def extract_func_from_class_node(class_node, no_super):
    func_list = class_node.body
    for func_node in func_list:
        if type(func_node) != ast.FunctionDef:
            continue

        if no_super and is_init_func(func_node):
            new_init_func_body = []
            for node in func_node.body:
                if is_super_expr(node):
                    continue
                new_init_func_body += [node]
            func_node.body = new_init_func_body

        arg_list = func_node.args.args
        new_arg_list = []
        for arg in arg_list:
            if arg.arg == "self" or arg.arg == "cls":
                continue
            new_arg_list += [arg]
        func_node.args.args = new_arg_list
    return func_list


def remove_class_from_ast(ast_tree, no_super):
    class_nodes = []
    for node in ast.walk(ast_tree):
        for idx, child in enumerate(ast.iter_child_nodes(node)):
            if type(child) == ast.ClassDef:
                child.parent = node
                if type(node) in [ast.If, ast.Try, ast.For]:
                    if child in node.body:
                        # it is in the if
                        child.idx = node.body.index(child)
                    elif child in node.orelse:
                        # it is in the else
                        child.idx = node.orelse.index(child)
                        child.is_else = True
                    elif child in node.finalbody:
                        child.idx = node.finalbody.index(child)
                        child.is_final = True
                    else:
                        raise (
                            f"Not in the body, another speciall case may happen, please look into this node: {ast.dump(node)}"
                        )
                else:
                    child.idx = node.body.index(child)

                class_nodes = [child] + class_nodes

    if len(class_nodes) == 0:
        # nothing to change
        return None

    for class_node in class_nodes:
        func_list = extract_func_from_class_node(class_node, no_super)
        idx = class_node.idx

        # addressing classes in the else condition
        if hasattr(class_node, "is_else") and class_node.is_else:
            class_node.parent.orelse.pop(idx)
            class_node.parent.orelse = (
                class_node.parent.orelse[:idx]
                + func_list
                + class_node.parent.orelse[idx:]
            )
        elif hasattr(class_node, "is_final") and class_node.is_final:
            class_node.parent.finalbody.pop(idx)
            class_node.parent.finalbody = (
                class_node.parent.finalbody[:idx]
                + func_list
                + class_node.parent.finalbody[idx:]
            )
        else:
            class_node.parent.body.pop(idx)
            class_node.parent.body = (
                class_node.parent.body[:idx]
                + func_list
                + class_node.parent.body[idx:]
            )

    return ast_tree


def remove_self_cls_str(script):
    return script.replace("self.", "").replace("cls.", "")


def remove_class(script, no_super=False):
    ast_tree = ast.parse(script)
    processed_ast_tree = remove_class_from_ast(ast_tree, no_super)
    if processed_ast_tree:
        return remove_self_cls_str(astunparse.unparse(processed_ast_tree))
    return None


# Docstring
def undocstring(source):
    try:
        parsed = ast.parse(source)

        for node in ast.walk(parsed):
            # print("Node is : ", node)
            # print("Node value is : ",node.body[0].value.s)

            if not isinstance(
                node,
                (
                    ast.Module,
                    ast.FunctionDef,
                    ast.ClassDef,
                    ast.AsyncFunctionDef,
                ),
            ):  # , ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef
                continue

            if not len(node.body):
                continue

            if not isinstance(node.body[0], ast.Expr):
                continue

            if not hasattr(node.body[0], "value") or not isinstance(
                node.body[0].value, ast.Str
            ):
                continue

            # Uncomment lines below if you want print what and where we are removing
            #
            #

            node.body = node.body[1:]

        class toLower(ast.NodeTransformer):
            def visit_arg(self, node):
                return ast.arg(**{**node.__dict__, "arg": node.arg.lower()})

            def visit_Name(self, node):
                # print("node id is : ",node.id)
                return ast.Name(**{**node.__dict__, "id": node.id.lower()})

        new_code = astunparse.unparse(parsed)  # toLower().visit(parsed))
        # print(new_code)
        return new_code
    except:
        parsed = None
        return parsed


# List Comp
def for_loop(text):
    def wrap_if(body, compare):
        return ast.If(test=compare, body=[body], orelse=[])

    def wrap_for(body, a, index):
        return ast.For(
            target=a.value.generators[index].target,
            iter=a.value.generators[index].iter,
            body=body,
            lineno=a.lineno + 1,
            orelse=[],
        )

    def comp_to_expl(tree):
        if hasattr(tree, "body"):
            i = 0
            while i < len(tree.body):
                if isinstance(a := tree.body[i], ast.Assign) and isinstance(
                    a.value, ast.ListComp
                ):
                    for_loop_body = [
                        ast.Expr(
                            value=ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(id=a.targets[0].id),
                                    attr="append",
                                    ctx=ast.Load(),
                                ),
                                args=[a.value.elt],
                                keywords=[],
                            )
                        )
                    ]

                    for ind, for_loop in enumerate(a.value.generators[::-1]):
                        ind = len(a.value.generators) - 1 - ind
                        for if_state in a.value.generators[ind].ifs:
                            for_loop_body = wrap_if(for_loop_body, if_state)

                        for_loop_body = wrap_for(for_loop_body, a, ind)

                    tree.body = (
                        tree.body[:i]
                        + [
                            ast.Assign(
                                targets=[ast.Name(id=a.targets[0].id)],
                                value=ast.List(elts=[]),
                                lineno=a.lineno,
                            )
                        ]
                        + [for_loop_body]
                        + tree.body[i + 1 :]
                    )
                    i += 1
                i += 1

        for i in getattr(tree, "_fields", []):
            if isinstance(v := getattr(tree, i, None), list):
                for i in v:
                    comp_to_expl(i)
            elif isinstance(v, ast.AST):
                comp_to_expl(v)

    try:
        parsed = ast.parse(text)
    except:
        return None

    try:
        comp_to_expl(parsed)
    except:
        pass

    return astunparse.unparse(parsed)


# Casing


class toLower(ast.NodeTransformer):
    def visit_arg(self, node):

        return ast.arg(
            **{**node.__dict__, "arg": node.arg.lower().replace("_", "")}
        )

    def visit_Name(self, node):
        return ast.Name(**{**node.__dict__, "id": node.id.lower()})


class toFuncLower(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        return ast.FunctionDef(
            **{**node.__dict__, "name": node.name.lower().replace("_", "")}
        )


def uncasing(script):
    final_code = None
    try:
        parsed = ast.parse(script)
        new_code = astunparse.unparse(toLower().visit(parsed))
        parsed_new = ast.parse(new_code)
        final_code = astunparse.unparse(toFuncLower().visit(parsed_new))
    except SyntaxError:
        final_code = None
    except RecursionError:
        final_code = None
    except TypeError:
        final_code = None

    return final_code


import typer

# source_dir = "data/evaluation_set.csv"
# output_dir =  "data/eval_parallel_corpora/eval_set_individual_feat.csv"
def main(source_dir, output_dir):
    eval_set_df = pd.read_csv(source_dir)

    uncommented_scripts = []
    no_class_scripts = []
    undocstring_scripts = []
    for_loop_scripts = []
    uncasing_scripts = []

    for idx, script in enumerate(tqdm(eval_set_df["content"])):
        uncommented_scripts += [uncomment(script)]
        try:
            if type(script) != str:
                raise (SyntaxError)
            no_class_scripts += [remove_class(script)]
        except SyntaxError:
            no_class_scripts += [None]
        except Exception as e:
            no_class_scripts += [None]
            print(script)
            print(e)
            raise (e)
        try:
            undocstring_scripts += [undocstring(script)]
        except Exception as e:
            undocstring_scripts += [None]

        try:
            for_loop_scripts += [for_loop(script)]
        except Exception as e:
            for_loop_scripts += [None]

        try:
            uncasing_scripts += [uncasing(script)]
        except Exception as e:
            uncasing_scripts += [None]

    eval_set_df["uncommented_content"] = uncommented_scripts
    eval_set_df["no_class_content"] = no_class_scripts
    eval_set_df["no_docstring_content"] = undocstring_scripts
    eval_set_df["no_comp_content"] = for_loop_scripts
    eval_set_df["no_casing_content"] = uncasing_scripts
    eval_set_df.to_csv(output_dir)


if __name__ == "__main__":
    typer.run(main)

