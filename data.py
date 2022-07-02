import numpy as np
from transformers import PLBartTokenizer
from transformers import default_data_collator
from datasets import load_from_disk, Dataset
from torch.utils.data import DataLoader

from config import BATCH_SIZE, PLBART_TRAIN, PLBART_TEST
from vocab import Vocab

train_dataset = load_from_disk(PLBART_TRAIN)
test_dataset = load_from_disk(PLBART_TEST)
train_dataset.set_format(columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(columns=["input_ids", "attention_mask", "labels"])

cluster_labels = np.array(train_dataset["labels"])
cluster_labels_no_outliers = cluster_labels[cluster_labels != -1]
cluster_vocab = Vocab([cluster_labels])
STYLE_DIM = len(cluster_vocab) - 1

# train_dataset = Dataset.from_dict(train_dataset[:64])
test_dataset = Dataset.from_dict(test_dataset[:64])

tokenizer = PLBartTokenizer.from_pretrained(
    "uclanlp/plbart-multi_task-python",
    language_codes="multi",
    src_lang="python",
    tgt_lang="python",
)


def get_data_loader(dataset, split="train"):
    data_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, collate_fn=default_data_collator
    )
    return data_loader

