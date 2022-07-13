import subprocess
import sys
import os

import torch
import pandas as pd


# configuration
NO_OUTLIERS = True
EXPERIMENT_NAME = "plbart_python_classifier_no_outliers"
CHECKPOINT = "uclanlp/plbart-multi_task-python"
# CHECKPOINT = "uclanlp/plbart-base"


# getting free GPU
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
    no_outlier_idx = [idx for idx, _bool in enumerate(train_plbart_dataset["labels"] != -1) if _bool]
    train_plbart_dataset = train_plbart_dataset.select(no_outlier_idx)
    
    no_outlier_idx = [idx for idx, _bool in enumerate(test_plbart_dataset["labels"] != -1) if _bool]
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

# train_plbart_dataset = Dataset.from_dict(train_plbart_dataset[:128])
# test_plbart_dataset = Dataset.from_dict(test_plbart_dataset[:128])

sorted_dataset = test_plbart_dataset.sort("labels")
sorted_dataset["labels"]

# sample and save data subset
torch.set_printoptions(threshold=10_000)
# sample_plbart = train_plbart_dataset.shuffle(seed=34).select(range(1000))
# sample_codet5 = train_codet5_dataset.shuffle(seed=34).select(range(1000))
# sample_plbart[0]
"""style_seperator = '[STY]'

sample = tokenizer(style_seperator, return_tensors="pt")
style_chars = []
for i in range (-1, 25):
    style_chars.append(f"<style{i}>")
print (style_chars)

#test_codet5_dataset.save_to_disk("datasets/plbart_sample_testing.hf")
#sample_codet5.save_to_disk("datasets/codet5_sample.hf")
example_python_phrase = "def maximum(a,b,c):NEW_LINE_INDENTreturn max([a,b,c])<style1>"
inputs = tokenizer(example_python_phrase, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits    
    print(logits)"""

from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    PLBartForSequenceClassification,
)

model = PLBartForSequenceClassification.from_pretrained(
    CHECKPOINT, num_labels=label_counts
)
# add classification layer

training_args = TrainingArguments(
    output_dir="test_trainer", evaluation_strategy="epoch"
)
metric = load_metric("accuracy")


def compute_metrics(
    eval_pred,
):  # this part prob wont work, parameter should be removed from trainer probably
    logits, labels = eval_pred

    predictions = np.argmax(logits[0], axis=-1)
    return metric.compute(predictions=predictions, references=labels)


from torch import nn
from transformers import Trainer


train_plbart_dataset = train_plbart_dataset.train_test_split(test_size=0.2)


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


training_args = TrainingArguments(
    # change the path cuz I got permission denied...
    output_dir=f"./results_ken/{EXPERIMENT_NAME}",
    learning_rate=1e-4,  # 2e-5, #
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=4,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    eval_accumulation_steps=100,
)
# debug_dataset =
print(training_args.device)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_plbart_dataset["train"],
    eval_dataset=train_plbart_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    # data_collator=data_collator,
)

trainer.train()
