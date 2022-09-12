import pandas as pd
from tqdm.auto import tqdm

from parallel_corpora_gen_script import (
    remove_class,
    for_loop,
    uncasing,
    undocstring,
)
from parallel_preprocessing_script import recover_fixed_docstring_labels
from utils.regex_parse import comment

# constants
feat_order = ["docstring", "list_comp", "class", "casing", "comment"]


def clean_df(current_df, left, right):
    current_df = current_df.dropna()
    current_df = current_df[current_df[left].notnull()]
    current_df = current_df[current_df[right].notnull()]
    current_df = current_df[current_df[left] != "Nan"]
    current_df = current_df.query(f"{left} != {right}")
    return current_df


def clean_full_df(df, target_feats):
    if "list_comp" in target_feats:
        df = clean_df(df, "no_comp_content", "uncommented_content")
    if "class" in target_feats:
        df = clean_df(df, "no_class_content", "uncommented_content")
    if "casing" in target_feats:
        df = clean_df(df, "no_casing_content", "uncommented_content")
    if "comment" in target_feats:
        df = clean_df(df, "uncommented_content", "content")
    return df


def feature_convert(
    current_df,
    feat,
    feat_func,
    first_feat,
    x_col,
    y_col,
    target_feats,
    csv_name,
):
    if first_feat == feat:
        df = pd.read_csv(csv_name)
        df = clean_full_df(df, target_feats)
        cols = [x_col, y_col]
        if y_col != "uncommented_content":
            cols += ["uncommented_content"]
        current_df = df[cols].copy().rename(columns={x_col: "X", y_col: "Y"})
        current_df = clean_df(
            current_df,
            "X",
            "uncommented_content" if y_col != "uncommented_content" else "Y",
        )
    else:
        processed_scripts = []
        for script in tqdm(current_df["X"], desc=feat):
            try:
                processed_scripts += [feat_func(script)]
            except:
                processed_scripts += [None]
        current_df["Z"] = current_df["X"].to_numpy()
        current_df["X"] = processed_scripts
        current_df = clean_df(current_df, "X", "Z")
    return current_df


import typer


def get_csv_dir(is_short):
    if is_short:
        return "data/eval_parallel_corpora/eval_set_short_individual_feat.csv"
    else:
        return "data/eval_parallel_corpora/eval_set_individual_feat.csv"


def main(target_feat, is_short=False):

    target_feats = target_feat.split("+")
    first_feat = None
    for f in target_feats:
        assert f in feat_order

    for feat in feat_order:
        if feat in target_feats:
            first_feat = feat
            break
    csv_dir = get_csv_dir(is_short)
    # combined generate pipeline
    Y_column_name = (
        "content" if "comment" in target_feats else "uncommented_content"
    )
    current_df = None
    ## docstring
    ### fix docstring length
    if "docstring" in target_feats:
        if is_short:
            current_df = feature_convert(
                current_df,
                "docstring",
                undocstring,
                first_feat,
                "no_docstring_content",
                Y_column_name,
                target_feats,
                csv_dir,
            )

        else:
            current_df = pd.read_csv(
                "data/eval_parallel_corpora/eval_set_fixed_docstring.csv"
            )

    ## list comp
    if "list_comp" in target_feats:
        current_df = feature_convert(
            current_df,
            "list_comp",
            for_loop,
            first_feat,
            "no_comp_content",
            Y_column_name,
            target_feats,
            csv_dir,
        )

    ## class
    if "class" in target_feats:
        current_df = feature_convert(
            current_df,
            "class",
            remove_class,
            first_feat,
            "no_class_content",
            Y_column_name,
            target_feats,
            csv_dir,
        )

    ## casing
    if "casing" in target_feats:
        current_df = feature_convert(
            current_df,
            "casing",
            uncasing,
            first_feat,
            "no_casing_content",
            Y_column_name,
            target_feats,
            csv_dir,
        )

    ## if docstring applied, recover comments in labels
    if (
        "docstring" in target_feats
        and "comment" in target_feats
        and not is_short
    ):
        current_df = recover_fixed_docstring_labels(current_df)

    ## comment
    ### check whether the label contains comments
    if "comment" in target_feats:
        commented_scripts = []
        for script in tqdm(current_df["Y"], desc="comment"):
            commented_scripts += [script if len(comment(script)) > 0 else None]
        current_df["Y"] = commented_scripts
        current_df = clean_df(current_df, "X", "Y")

    current_df = (
        current_df[["X", "Y"]]
        .copy()
        .rename(columns={"X": "input", "Y": "label"})
    )
    suffix = "_".join([feat for feat in feat_order if feat in target_feats])
    if is_short:
        suffix = f"short_{suffix}"
    current_df.to_csv(f"data/eval_parallel_corpora/eval_set_{suffix}.csv")


if __name__ == "__main__":
    typer.run(main)

