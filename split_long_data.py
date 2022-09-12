import ast
import astunparse
import difflib
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_from_disk

max_raw_length = 1250


def traverse_whole_tree(root, visited_set):
    for node in ast.walk(root):
        visited_set.add(node)


def is_exceed_bound(script):
    return len(script) > max_raw_length


def extract_class_methods(source: str):
    short_codes = []
    full_codes = []
    visited_set = set()
    try:
        parsed = ast.parse(source)
        for node in ast.walk(parsed):
            if node in visited_set:
                continue
            if not isinstance(
                node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)
            ):  # , ast.Module
                continue
            if not len(node.body):
                continue

            script = astunparse.unparse(node)

            if is_exceed_bound(script):
                continue

            traverse_whole_tree(node, visited_set)

            short_codes += [script]
            full_codes += [source]

        return short_codes, full_codes
    except:
        return short_codes, full_codes


def recover_fixed_docstring_labels(short_codes, full_codes):
    """recover the label which contains comments that were removed during fixing docstring"""

    recovered_labels = []

    for idx, code in tqdm(
        enumerate(short_codes),
        total=len(short_codes),
        desc="recover_fixed_docstring_labels",
    ):

        str_1 = code
        str_2 = full_codes[idx]

        s1 = str_1.split("\n")
        s2 = str_2.split("\n")

        # get the first and last line of the cropped text
        first_line = ""
        for line in s1:
            if len(line.split()) > 0:
                first_line = line
                break

        last_line = ""
        for line in reversed(s1):
            if len(line.split()) > 0:
                last_line = line
                break

        start_score = -1
        end_score = -1
        start_idx_cands = []
        end_idx_cands = []
        # match the first and last line to every line in the full text
        for idx, line in enumerate(s2):
            # take the one with the largest score as the start/end line index candidates
            start_matcher = difflib.SequenceMatcher(a=first_line, b=s2[idx])
            score = start_matcher.ratio()
            if score > start_score:
                # if larger score appear then reset candidates
                start_score = score
                start_idx_cands = [idx]
            elif score == start_score:
                # if equal, make it an another candidate
                start_idx_cands += [idx]

            end_matcher = difflib.SequenceMatcher(a=last_line, b=s2[idx])
            score = end_matcher.ratio()
            if score > end_score:
                # if larger score appear then reset candidates
                end_score = score
                end_idx_cands = [idx]
            elif score == end_score:
                # if equal, make it an another candidate
                end_idx_cands += [idx]

        target_len = len(str_1)
        target_text = None
        min_diff = float("inf")
        # permutate over all start and end combinations
        for start_idx in start_idx_cands:
            for end_idx in end_idx_cands:
                if start_idx > end_idx:
                    continue

                # use the length difference between candidate text(cropped from full text with comments) and the target text(cropped text without comment)
                # as the score(smaller is better, minimum is 0)
                cand_text = "\n".join(s2[start_idx : end_idx + 1])
                cand_len = len(cand_text)
                diff_len = abs(target_len - cand_len)

                # the smaller one means it has same size as the target, which may be what we want
                if diff_len < min_diff:
                    min_diff = diff_len
                    target_text = cand_text

        recovered_labels += [target_text]

    return recovered_labels


from sklearn.model_selection import train_test_split

# processing long script
def process_long_scripts(long_scripts, sample_p):
    short_codes = []
    full_codes = []

    long_scripts = train_test_split(long_scripts, test_size=sample_p)[1]

    for script in tqdm(long_scripts, desc="processing long scripts"):
        short_code_list, full_code_list = extract_class_methods(script)
        short_codes += short_code_list
        full_codes += full_code_list

    recovered_short_codes = recover_fixed_docstring_labels(
        short_codes, full_codes
    )
    return recovered_short_codes


def split_long_short(df, tok_lens_hf):
    long_scripts = []
    short_scripts = []

    for idx, script in tqdm(
        enumerate(df["content"]), total=len(df), desc="splitting data"
    ):
        # make sure there is enough space for prompt tokens
        if tok_lens_hf[idx]["length"] <= 480:
            short_scripts += [script]
        else:
            long_scripts += [script]

    return long_scripts, short_scripts


import typer

input_dir_dict = {
    "train": "data/labeled_code/bq_data_outlier.csv",
    "eval": "data/evaluation_set.csv",
}

output_dir_long_dict = {
    "train": "data/labeled_code/bq_data_outlier_long.csv",
    "eval": "data/evaluation_set_long.csv",
}
output_dir_short_dict = {
    "train": "data/labeled_code/bq_data_outlier_short.csv",
    "eval": "data/evaluation_set_short.csv",
}

len_hf_dir_dict = {
    "train": "datasets/train_raw_scripts_dataset.hf",
    "eval": "datasets/evaluation_dataset/raw_scripts_dataset.hf",
}


def main(split):
    assert split in ["train", "eval"]

    input_df = pd.read_csv(input_dir_dict[split])
    input_df = input_df[["content"]]

    len_hf = load_from_disk(len_hf_dir_dict[split])
    long_scripts, short_scripts = split_long_short(input_df, len_hf)
    # short_df = pd.DataFrame({"content": short_scripts})
    # short_df.to_csv(output_dir_short_dict[split])

    downsize_p = (len(short_scripts) / 7) / len(long_scripts)
    long_scripts = process_long_scripts(long_scripts, downsize_p)

    long_df = pd.DataFrame({"content": long_scripts})
    long_df.to_csv(output_dir_long_dict[split])


if __name__ == "__main__":
    typer.run(main)
