import numpy as np
from transformers import PLBartTokenizer, RobertaTokenizer
from transformers import default_data_collator
from datasets import load_from_disk, Dataset
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    PLBART_TRAIN,
    PLBART_TEST,
    CODET5_TRAIN,
    CODET5_TEST,
    MODEL,
)
from vocab import Vocab

train_path = PLBART_TRAIN if MODEL == "plbart" else CODET5_TRAIN
test_path = PLBART_TEST if MODEL == "plbart" else CODET5_TEST
train_dataset = load_from_disk(train_path)
test_dataset = load_from_disk(test_path)

train_dataset.set_format(columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(columns=["input_ids", "attention_mask", "labels"])

train_cluster_labels = np.array(train_dataset["labels"])
test_cluster_labels = np.array(test_dataset["labels"])
# small set of code
train_indice_list = np.logical_or(
    # (train_cluster_labels == 0), (train_cluster_labels == 1) # small set
    train_cluster_labels != -1,
    train_cluster_labels != -1,  # no outliers
)
test_indice_list = np.logical_or(
    # (test_cluster_labels == 0), (test_cluster_labels == 1)
    test_cluster_labels != -1,
    test_cluster_labels != -1,  # no outliers
)
train_dataset = train_dataset.select(np.where(train_indice_list)[0])
test_dataset = test_dataset.select(np.where(test_indice_list)[0])
# train_dataset = Dataset.from_dict(train_dataset[:64])
# test_dataset = Dataset.from_dict(test_dataset[:64])


cluster_labels = np.array(train_dataset["labels"])
cluster_labels_no_outliers = cluster_labels[cluster_labels != -1]
cluster_vocab = Vocab([cluster_labels])
STYLE_DIM = len(np.unique(cluster_labels_no_outliers))


if MODEL == "codet5":
    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")
elif MODEL == "plbart":
    tokenizer = PLBartTokenizer.from_pretrained(
        "uclanlp/plbart-multi_task-python",
        language_codes="multi",
        src_lang="python",
        tgt_lang="python",
    )
else:
    raise ValueError("The model should be 'codet5' or 'plbart'")

def get_dataset(model_type, split="train"):
    train_path = PLBART_TRAIN if model_type == "plbart" else CODET5_TRAIN
    test_path = PLBART_TEST if model_type == "plbart" else CODET5_TEST
    assert split in ["train", "test"]
    
    path = train_path if split == "train" else test_path
    dataset = load_from_disk(path)

    dataset.set_format(columns=["input_ids", "attention_mask", "labels"])
    c_labels = train_cluster_labels if split == "train" else test_cluster_labels
    indice_list = c_labels != -1
    dataset = dataset.select(np.where(indice_list)[0])
    
    return dataset
    
    
def get_data_loader(dataset, split="train"):
    data_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, collate_fn=default_data_collator
    )
    return data_loader

