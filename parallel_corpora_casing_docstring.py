import ast
import astor

parsed = ast.parse(open('source.py').read())
for node in ast.walk(parsed):
    if not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
        continue

    if not len(node.body):
        continue

    if not isinstance(node.body[0], ast.Expr):
        continue

    if not hasattr(node.body[0], 'value') or not isinstance(node.body[0].value, ast.Str):
        continue

    # Uncomment lines below if you want print what and where we are removing
    # print("Node is : ",node)
    # print("Node value is : ",node.body[0].value.s)

    node.body = node.body[1:]

print('***** Processed source code output ******\n=========================================')

#print(astor.to_source(parsed))



class toLower(ast.NodeTransformer):

    def visit_arg(self, node):
        print("Inside args")
        return ast.arg(**{**node.__dict__, 'arg':node.arg.lower().replace("_","")})
    def visit_Name(self, node):
        print("Inside name")
        return ast.Name(**{**node.__dict__, 'id':node.id.lower()})
    

new_code = ast.unparse(toLower().visit(parsed))
#print(new_code)

class toFuncLower(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        print("Inside Func Def")
        return ast.FunctionDef(**{**node.__dict__, 'name':node.name.lower().replace("_","")})

parsed_new = ast.parse(new_code)
final_code = ast.unparse(toFuncLower().visit(parsed_new))
print(final_code)