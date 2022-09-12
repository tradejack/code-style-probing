import py_compile
import ast

from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

import os


bq_outlier_df = pd.read_csv("data/labeled_code/bq_data_outlier.csv")

data_dir_prefix = (
    "/data/users/team2_capstone/code-style-probing/data/BigQuery/raw_files"
)

# compilable eval
def is_compilable(input_code, input_type="text"):
    assert input_type in ["text", "file"]

    path = "./temp.py"
    if input_type == "text":
        with open(temp, "w") as f:
            f.write(input_code)
    if input_type == "file":
        path = input_code

    try:
        py_compile.compile(path, doraise=True)
    except py_compile.PyCompileError:
        return False
    except Exception as e:
        print(e)
        raise (e)

    return True


# parsable eval
def is_parsable(input_code):
    try:
        ast.parse(input_code)
    except SyntaxError:
        return False
    except Exception as e:
        print(input_code)
        print(e)
        raise (e)
    return True


compile_bools = []
for index, row in tqdm(
    bq_outlier_df[["repository", "filepath", "content"]].iterrows(),
    total=len(bq_outlier_df),
):
    repo = row["repository"]
    file_dir = row["filepath"]
    script = row["content"]

    path = f"{data_dir_prefix}/{repo}/{file_dir}"
    compile_bools += [is_compilable(path, input_type="file")]

parse_bools = []
for index, row in tqdm(
    bq_outlier_df[["content"]].iterrows(), total=len(bq_outlier_df)
):
    script = row["content"]
    parse_bools += [is_parsable(script)]

data = {"is_compilable": compile_bools, "is_parsable": parse_bools}
df = pd.DataFrame(data)
df.to_csv("data/bq_data_outlier_functionality.csv", index=False)
