import pandas as pd
from tqdm.auto import tqdm


def read_file_to_string(filename):
    f = open(filename, "rb")
    s = ""
    try:
        s = f.read()
    except:
        print(filename)
    f.close()
    return s.decode(errors="replace")


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


def calculate_ratio(numerator, denominator, round_val=6):
    return round(numerator / denominator, round_val) if denominator > 0 else 0


def metric_dict_to_df(metric_dict_list):
    return pd.DataFrame(metric_dict_list).fillna(0)


def print_split_line(s):
    print(f"\n====================={s.upper()}=====================\n")