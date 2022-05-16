import ast


def ast_parse(code):
    """
    Tracks:
        Docstring raw count, char length, word length, function density
        Variable  raw count, char length, casing, word length (if possible)
    """
    parse = ast.parse(code)
    return parse


if __name__ == "__main__":
    from helper import read_file_to_string

    test_code = read_file_to_string("./test_code.py")
    print(ast_parse(test_code))
