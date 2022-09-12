from pathlib import Path
from enum import Enum

import numpy as np
from datasets import load_from_disk
from transformers import *

fname_prefix = "/data/users/cting3/CodeNest/code-style-probing/"

# control codes
tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")


feat_order = ["docstring", "list_comp", "class", "casing", "comment"]
max_size = 512

# ver 1
def control_toks_add(example, control_toks):
    # takes a tokenized data set and adds the control sequence to the code
    input = example["input_ids"].copy()
    mask = example["attention_mask"].copy()
    idx_last = int(input.index(2))
    control_toks_len = len(control_toks["input_ids"]) - 2
    toks = control_toks["input_ids"][1:-1]
    free_space = max_size - idx_last - 1

    if free_space >= control_toks_len:
        toks = toks + [2]
        for idx in range(control_toks_len):
            input[idx_last + idx] = toks[idx]
            mask[idx_last + idx] = 1
    else:
        toks = toks + [2]
        start_idx = 511 - control_toks_len
        for idx in range(control_toks_len):
            input[start_idx + idx] = toks[idx]
            mask[start_idx + idx] = 1

    example["attention_mask"] = mask
    example["input_ids"] = input
    return example


# ver 2
def control_nl_toks_add(example, control_toks):
    # takes a tokenized data set and adds the control sequence to the code
    input = example["input_ids"].copy()
    mask = example["attention_mask"].copy()

    idx_last = int(input.index(2))

    control_toks_len = len(control_toks["input_ids"])
    free_space = max_size - idx_last - 1

    true_seq = input[: idx_last + 1]
    if free_space >= control_toks_len:
        input[:control_toks_len] = control_toks["input_ids"]
        input[control_toks_len : control_toks_len + len(true_seq)] = true_seq
        mask = np.array(mask)
        mask[: control_toks_len + idx_last] = 1
    else:
        input[:control_toks_len] = control_toks["input_ids"]
        fit_len = max_size - control_toks_len - 1
        input[control_toks_len:] = true_seq[:fit_len] + [2]
        mask = np.array(mask)
        mask[:] = 1

    example["attention_mask"] = list(mask)
    example["input_ids"] = input
    return example


def get_prompt(text):
    return f"<nl> {text} </nl>"


def transform_type_1(dataset, target_feats, target_path):
    transformation_dict = {feat: feat for feat in feat_order}
    transformation_dict["list_comp"] = "comp"
    transformation_dict["casing"] = "case"

    control_strs = []
    for feat in target_feats:
        transformation = transformation_dict[feat]
        control_strs += [f"# {transformation}"]

    control_str = "\n".join(control_strs)
    control_toks = tokenizer(control_str)
    control_tok_dataset = dataset.map(
        lambda ex: control_toks_add(ex, control_toks)
    )  # add extra args
    control_tok_dataset.save_to_disk(target_path)


def transform_type_2(dataset, target_feats, target_path):
    transformation_dict_type_2 = {
        "comment": "add comment",
        "class": "add class",
        "docstring": "add docstring",
        "casing": "change identifier casing",
        "list_comp": "change for loop to list comprehension",
    }

    control_strs = []
    for feat in target_feats:
        transformation = transformation_dict_type_2[feat]
        control_strs += [transformation]
    control_str = " , ".join(control_strs)
    prompt = get_prompt(control_str)
    control_toks = tokenizer(prompt, add_special_tokens=False)
    control_tok_dataset = dataset.map(
        lambda ex: control_nl_toks_add(ex, control_toks)
    )  # add extra args
    control_tok_dataset.save_to_disk(target_path)


def transform_type_3(dataset, target_feats, target_path):
    tokenizer.add_special_tokens(["<nl>", "</nl>"])
    transformation_dict_type_2 = {
        "comment": "add comment",
        "class": "add class",
        "docstring": "add docstring",
        "casing": "change identifier casing",
        "list_comp": "change for loop to list comprehension",
    }

    control_strs = []
    for feat in target_feats:
        transformation = transformation_dict_type_2[feat]
        control_strs += [transformation]
    control_str = " , ".join(control_strs)
    prompt = get_prompt(control_str)
    control_toks = tokenizer(prompt, add_special_tokens=False)
    control_tok_dataset = dataset.map(
        lambda ex: control_nl_toks_add(ex, control_toks)
    )  # add extra args
    control_tok_dataset.save_to_disk(target_path)


def transform_type_4(dataset, target_feats, target_path):
    """new prompt: simpler prompt"""
    transformation_dict_type_2 = {
        "comment": "comment",
        "class": "class",
        "docstring": "docstring",
        "casing": "casing",
        "list_comp": "list comprehension",
    }

    control_strs = []
    for feat in target_feats:
        transformation = transformation_dict_type_2[feat]
        control_strs += [transformation]
    control_str = "add " + " , ".join(control_strs)
    prompt = get_prompt(control_str)
    control_toks = tokenizer(prompt, add_special_tokens=False)
    control_tok_dataset = dataset.map(
        lambda ex: control_nl_toks_add(ex, control_toks)
    )  # add extra args
    control_tok_dataset.save_to_disk(target_path)


import typer


def main(
    source_dataset: Path,
    target_path: Path,
    target_feat: str,
    transformation_type: int,
):
    target_feats = target_feat.split("+")

    for f in target_feats:
        assert f in feat_order

    target_feats = [feat for feat in feat_order if feat in target_feats]

    dataset = load_from_disk(source_dataset)
    if transformation_type == 1:
        transform_type_1(dataset, target_feats, target_path)

    if transformation_type == 2:
        transform_type_2(dataset, target_feats, target_path)

    if transformation_type == 3:
        transform_type_3(dataset, target_feats, target_path)

    if transformation_type == 4:
        transform_type_4(dataset, target_feats, target_path)


if __name__ == "__main__":
    typer.run(main)
