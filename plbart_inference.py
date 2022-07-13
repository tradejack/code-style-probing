# last_ckpt = "results_ken/plbart_classifier/checkpoint-172000/trainer_state.json"


EXPERIMENT_NAME = "plbart_python_classifier_no_outliers"
last_ckpt = f"results_ken/{EXPERIMENT_NAME}/checkpoint-33596"
NO_OUTLIERS = True

# import json
# state = json.load(open(last_ckpt))
# epochs = [log["epoch"] for log in state["log_history"] if log["epoch"].is_integer()]
# losses = [log["loss"] for log in state["log_history"] if log["epoch"].is_integer()]
# steps = [log["step"] for log in state["log_history"] if log["epoch"].is_integer()]
# ckpt_filenames = [f"results_ken/plbart_classifier/checkpoint-{step}" for step in steps]

import subprocess
import sys
import os

import torch
import pandas as pd


if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO


def get_free_gpu():
    gpu_stats = subprocess.check_output(
        ["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]
    )
    gpu_df = pd.read_csv(
        StringIO(gpu_stats.decode("utf-8")),
        names=["memory.used", "memory.free"],
        skiprows=1,
    )
    print("GPU usage:\n{}".format(gpu_df))
    gpu_df["memory.free"] = gpu_df["memory.free"].map(
        lambda x: int(x.rstrip(" MiB"))
    )
    idx = gpu_df["memory.free"].idxmax()
    print(
        "Returning GPU{} with {} free MiB".format(
            idx, gpu_df.iloc[idx]["memory.free"]
        )
    )
    return idx


free_gpu_id = get_free_gpu()
print(free_gpu_id)

# !pip list
import os

# print (os.environ["CUDA_VISIBLE_DEVICES"])
os.environ["CUDA_VISIBLE_DEVICES"] = str(free_gpu_id)
print(os.environ["CUDA_VISIBLE_DEVICES"])
# import model file
import torch

import numpy as np

# import pandas as pd
from transformers import *
import torch
import torch.nn as nn
import torch.nn.functional as F

# from torch.utils.data import Dataset
from datasets import load_from_disk, load_metric, Dataset


train_plbart_dataset = load_from_disk("datasets/plbart_train.hf")
test_plbart_dataset = load_from_disk("datasets/plbart_test.hf")
# train_plbart_dataset = Dataset.from_dict(train_plbart_dataset[4200:4300])
# test_plbart_dataset = Dataset.from_dict(test_plbart_dataset[4200:4300])
train_plbart_dataset.set_format(
    type="np", columns=["input_ids", "attention_mask", "labels"]
)
test_plbart_dataset.set_format(
    type="np", columns=["input_ids", "attention_mask", "labels"]
)


label_counts = 27

if NO_OUTLIERS:
    no_outlier_idx = [
        idx
        for idx, _bool in enumerate(train_plbart_dataset["labels"] != -1)
        if _bool
    ]
    train_plbart_dataset = train_plbart_dataset.select(no_outlier_idx)

    no_outlier_idx = [
        idx
        for idx, _bool in enumerate(test_plbart_dataset["labels"] != -1)
        if _bool
    ]
    test_plbart_dataset = test_plbart_dataset.select(no_outlier_idx)

    label_counts -= 1

tokenizer = PLBartTokenizer.from_pretrained(
    "uclanlp/plbart-base", src_lang="python", tgt_lang="python"
)


unk_id = tokenizer.convert_tokens_to_ids("<unk>")


def preprocess(example):
    # print (label['labels'])
    if NO_OUTLIERS == False:
        # replace negative by adding all labels by 1
        example["labels"] = example["labels"] + 1

    # remove duplicate eos token
    eos_mask = example["input_ids"] == 2
    if len(torch.unique_consecutive(torch.Tensor(eos_mask).sum(1))) > 1:
        invalid_idx = [
            idx for idx, mask in enumerate(eos_mask) if mask.sum() > 1
        ]
        # print(invalid_idx)
        for idx in invalid_idx:
            invalid_token_idx = [
                idx for idx, mask in enumerate(eos_mask[idx]) if mask == 1
            ]
            if len(invalid_token_idx) < 1:
                continue
            for invalid_token_pos in range(len(invalid_token_idx) - 1):
                # print(tokenizer.batch_decode([example["input_ids"][idx]]))
                example["input_ids"][idx][
                    invalid_token_idx[invalid_token_pos]
                ] = unk_id
                # print(tokenizer.batch_decode([example["input_ids"][idx]]))

    return example


train_plbart_dataset = train_plbart_dataset.map(preprocess, batched=True)
test_plbart_dataset = test_plbart_dataset.map(preprocess, batched=True)
train_plbart_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"]
)
test_plbart_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"]
)


train_plbart_dataset = train_plbart_dataset.train_test_split(test_size=0.01)


metric = load_metric("accuracy")


def compute_metrics(
    eval_pred,
):  # this part prob wont work, parameter should be removed from trainer probably
    logits, labels = eval_pred

    predictions = np.argmax(logits[0], axis=-1)
    return metric.compute(predictions=predictions, references=labels)


from torch import nn
from transformers import Trainer


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, 2.0, 3.0])
        )  # calculate weights
        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels), labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss


from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    PLBartForSequenceClassification,
)

training_args = TrainingArguments(
    # change the path cuz I got permission denied...
    output_dir=f"./results_ken/{EXPERIMENT_NAME}_inference",
    learning_rate=1e-4,  # 2e-5, #
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    eval_accumulation_steps=100,
)

model = PLBartForSequenceClassification.from_pretrained(
    last_ckpt, num_labels=label_counts
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_plbart_dataset["train"],
    eval_dataset=train_plbart_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    # data_collator=data_collator,
)

preds = trainer.predict(test_plbart_dataset)


pred_labels = preds.predictions[0].argmax(-1)
pred_labels_df = pd.DataFrame(pred_labels)
pred_labels_df.to_csv(f"plbart_preds_{EXPERIMENT_NAME}.csv")

