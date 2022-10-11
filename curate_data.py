import os
import logging

import pandas as pd
from datasets import load_from_disk


from combined_parallel_gen_script import clean_full_df
from tokenize_raw_script import raw_tokenize

logging.basicConfig(level=logging.INFO)

DATA_DIR = "/data/ken/data/code"
EVAL_IND_FEAT_PATH = f"{DATA_DIR}/eval_set_individual_feat.csv"

logging.info("Loading data")
# load the data: evaluation short(raw tokenization has done?)
eval_df = pd.read_csv(EVAL_IND_FEAT_PATH)
# add an original index column for mapping back to the original data
eval_df["original_idx"] = list(range(len(eval_df)))


def get_untokenized_df(df, visited_set):
    return df.loc[~df["original_idx"].isin(visited_set)]


def get_token_length_dataset(path):
    dataset = load_from_disk(path)
    dataset = dataset.remove_columns(["content", "input_ids", "attention_mask"])
    dataset = dataset.rename_column("__index_level_0__", "original_idx")
    return dataset


feats = ["list_comp", "class", "comment", "docstring", "casing"]

visited_set = set()

length_df = pd.DataFrame({})

for feat in feats:
    # filter out the non-transform data
    logging.info(f"Cleaning {feat}")
    df = clean_full_df(eval_df.copy(), target_feats=[feat])
    print(df)

    # raw tokenization
    logging.info(f"Tokenizing {feat}")
    untokenized_df = get_untokenized_df(df, visited_set)

    # clean Nan
    untokenized_df = untokenized_df[untokenized_df["content"] != "Nan"].dropna(
        subset=["content"]
    )

    print(untokenized_df)
    if len(untokenized_df) > 0:
        visited_set.update(untokenized_df["original_idx"].tolist())
        hf_path = f"{DATA_DIR}/{feat}.hf"
        if not os.path.exists(hf_path):
            _ = raw_tokenize(untokenized_df, hf_path)

        # concat all the length data
        logging.info(f"Getting Token Length Dataset of {feat}")
        dataset = get_token_length_dataset(f"{DATA_DIR}/{feat}.hf")
        length_df = pd.concat([length_df, pd.DataFrame(dataset)])

# sort by idx and output as csv
logging.info("Saving Length Data Frame")
length_df = length_df.sort_values(by=["original_idx"])
length_df.to_csv(f"{DATA_DIR}/eval_set_length.csv")
