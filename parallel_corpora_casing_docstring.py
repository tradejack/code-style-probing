import ast
import astor

parsed = ast.parse(open("source.py").read())
for node in ast.walk(parsed):
    # let's work only on functions & classes definitions
    if isinstance(node, (ast.FunctionDef)):
        print("Yes")
    if not isinstance(
        node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)
    ):
        # print("Inside If 1")
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
    # print("Node is : ",node)
    # print("Node value is : ",node.body[0].value.s)

    node.body = node.body[1:]

print(
    "***** Processed source code output ******\n========================================="
)

# print(astor.to_source(parsed))


class toLower(ast.NodeTransformer):
    def visit_arg(self, node):
        print(node.lineno)
        print("node arg is : ", node.arg)
        return ast.arg(**{**node.__dict__, "arg": node.arg.lower()})

    def visit_Name(self, node):
        print("node id is : ", node.id)
        return ast.Name(**{**node.__dict__, "id": node.id.lower()})


# new_code = ast.unparse(toLower().visit(ast.parse(parsed)))
new_code = ast.unparse(toLower().visit(parsed))
print(new_code)


# def count_method_with_docstring(ast_tree):
#     count = 0
#     for node in ast.walk(ast_tree):
#         _id = type(node)

#         if _id == ast.FunctionDef:
#             ds = node.name
#             print(ds.lower())

#     return ds.lower()


# abc = count_method_with_docstring(parsed)
# #print(abc)
