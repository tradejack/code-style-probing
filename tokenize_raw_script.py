import pandas as pd
from datasets import Dataset
from transformers import RobertaTokenizer


import typer

input_dir_dict = {
    "train": "data/labeled_code/bq_data_outlier.csv",
    "eval": "data/evaluation_set.csv",
}
output_dir_dict = {
    "train": "datasets/train_raw_scripts_dataset.hf",
    "eval": "datasets/evaluation_dataset/raw_scripts_dataset.hf",
}


def main(split):
    assert split in ["train", "eval"]

    input_df = pd.read_csv(input_dir_dict[split])
    output_dir = output_dir_dict[split]

    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")

    def tokenization(example):
        return_dict = tokenizer(example["content"])
        return_dict["length"] = [len(ids) for ids in return_dict["input_ids"]]
        return return_dict

    current_df = input_df[["content"]]
    new_df = current_df.copy()
    for idx, script in enumerate(current_df["content"]):
        if type(script) != str:
            new_df["content"][idx] = ""
    current_df = new_df

    dataset = Dataset.from_pandas(current_df)
    dataset = dataset.map(tokenization, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    dataset.save_to_disk(output_dir)


if __name__ == "__main__":
    typer.run(main)
