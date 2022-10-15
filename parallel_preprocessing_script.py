if __name__ == "__main__":
    import sys

    assert len(sys.argv) > 1
import ast
import astunparse
import difflib
import pandas as pd
from tqdm.auto import tqdm


feat_order = [
    "docstring",
    "list_comp",
    "class",
    "casing",
    "comment",
    "decorator",
]
# Docstring fix length
def undocstring(source):
    if '"""' in source or "'''" in source:
        try:
            parsed = ast.parse(source)
            for node in ast.walk(parsed):
                # print("Node value is : ",node.body[0].value.s)

                if not isinstance(
                    node,
                    (
                        ast.Module,
                        ast.FunctionDef,
                        ast.ClassDef,
                        ast.AsyncFunctionDef,
                    ),
                ):
                    continue

                if not len(node.body):
                    continue

                if not isinstance(node.body[0], ast.Expr):
                    continue

                if not hasattr(node.body[0], "value") or not isinstance(
                    node.body[0].value, ast.Str
                ):
                    continue

                node.body = node.body[1:]
                new_code = astunparse.unparse(
                    parsed
                )  # toLower().visit(parsed))
                # print(new_code)
                return new_code
        except:
            return None
    return source


def undocstring_method(source):
    try:
        parsed = ast.parse(source)
        return_dict = {"inputs": [], "labels": []}
        for node in ast.walk(parsed):
            # print("Node is : ",node)
            # print("Node value is : ",node.body[0].value.s)

            if not isinstance(
                node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)
            ):  # , ast.Module
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
            return_dict["labels"].append(astunparse.unparse(node))
            node.body = node.body[1:]
            return_dict["inputs"].append(astunparse.unparse(node))

        return return_dict
    except:
        return {"inputs": [], "labels": []}


# split docstring_df for short data items and long data items
def len_content(source):
    return len(source)


# extract methods from long_df with docstrings from uncommented no docstring input and original method as
def method_docstring(source):
    methods = undocstring_method(source)
    inputs = methods["inputs"]
    labels = methods["labels"]
    return inputs, labels


def fix_docstring_len(current_df):
    current_df = current_df[current_df["X"].notnull()]
    current_df = current_df[current_df["Y"].notnull()]
    current_df["lens"] = current_df["Y"].apply(len_content)
    short_df = current_df[current_df["lens"] < 2000]
    long_df = current_df[current_df["lens"] >= 2000]

    repaired_df = short_df.query("X != Y")

    new_no_docstring_scripts = []
    for script in tqdm(repaired_df["X"], desc="short_df"):
        new_no_docstring_scripts += [undocstring(script)]
    repaired_df["X"] = new_no_docstring_scripts
    repaired_df = repaired_df[repaired_df["X"].notnull()]

    inputs = []
    labels = []
    full_labels = []
    for script in tqdm(long_df["Y"], desc="long_df"):
        doc_method = method_docstring(script)
        for idx in range(len(doc_method[0])):
            inputs += [doc_method[0][idx]]
            labels += [doc_method[1][idx]]
            full_labels += [script]

    # combine short df and long df fixes and save to file
    inputs.extend(list(repaired_df["X"]))
    labels.extend(list(repaired_df["Y"]))
    full_labels.extend(list(repaired_df["Y"]))
    current_df = pd.DataFrame(
        {"X": inputs, "Y": labels, "Y_full": full_labels}
    )
    return current_df


def recover_fixed_docstring_labels(current_df):
    """recover the label which contains comments that were removed during fixing docstring"""
    new_df = current_df.copy()

    recovered_labels = []

    # examples in long_df
    long_df_bool = new_df["Y"] != new_df["Y_full"]

    for idx, row in tqdm(
        new_df[long_df_bool].iterrows(),
        total=len(new_df[long_df_bool]),
        desc="recover_fixed_docstring_labels",
    ):

        str_1 = row["Y"]
        str_2 = row["Y_full"]

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

    new_df[long_df_bool]["Y"] = recovered_labels

    current_df = new_df
    return current_df


import typer

# csv_fname="data/eval_parallel_corpora/eval_set_individual_feat.csv"
# output_dir="datasets/evaluation_dataset/codet5_{target_feat}_eval_set_padded.hf"


def main(target_feat, csv_fname, output_dir, is_short: bool = False):

    feat_names = [
        "comment",
        "class",
        "docstring",
        "list_comp",
        "casing",
        "decorator",
    ]

    if "+" in target_feat:
        # combined feats
        target_feats = target_feat.split("+")

        for f in target_feats:
            assert f in feat_order

        target_feat = "_".join(
            [feat for feat in feat_order if feat in target_feats]
        )
    else:
        assert target_feat in feat_names + ["all_feat"]

    from datasets import Dataset

    df = pd.read_csv(csv_fname)

    current_df = None
    if target_feat == "comment":
        current_df = (
            df[["content", "uncommented_content"]]
            .copy()
            .rename(columns={"uncommented_content": "X", "content": "Y"})
        )
    if target_feat == "class":
        current_df = (
            df[["uncommented_content", "no_class_content"]]
            .copy()
            .rename(
                columns={"no_class_content": "X", "uncommented_content": "Y"}
            )
        )
    if target_feat == "docstring":
        current_df = (
            df[["uncommented_content", "no_docstring_content"]]
            .copy()
            .rename(
                columns={
                    "no_docstring_content": "X",
                    "uncommented_content": "Y",
                }
            )
        )
    if target_feat == "list_comp":
        current_df = (
            df[["uncommented_content", "no_comp_content"]]
            .copy()
            .rename(
                columns={"no_comp_content": "X", "uncommented_content": "Y"}
            )
        )
    if target_feat == "casing":
        current_df = (
            df[["uncommented_content", "no_casing_content"]]
            .copy()
            .rename(
                columns={"no_casing_content": "X", "uncommented_content": "Y"}
            )
        )
    if target_feat == "decorator":
        current_df = (
            df[["uncommented_content", "no_decorator_content"]]
            .copy()
            .rename(
                columns={
                    "no_decorator_content": "X",
                    "uncommented_content": "Y",
                }
            )
        )

    if target_feat == "docstring" and not is_short:
        current_df = fix_docstring_len(current_df)

    if target_feat not in feat_names:
        current_df = df.copy().rename(columns={"input": "X", "label": "Y"})

    current_df = current_df[current_df["X"].notnull()]
    current_df = current_df[current_df["Y"].notnull()]
    current_df = current_df.query(f"X != Y")

    from transformers import RobertaTokenizer

    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")

    def tokenization(example):
        return_dict = tokenizer(
            example["X"], padding="max_length", truncation=True
        )
        labels = tokenizer(
            example["Y"], padding="max_length", truncation=True
        ).input_ids
        return_dict["labels"] = labels

        return return_dict

    dataset = Dataset.from_pandas(current_df)
    dataset = dataset.map(tokenization, batched=True)
    dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    dataset.format["type"]
    dataset.save_to_disk(output_dir)


if __name__ == "__main__":
    typer.run(main)
