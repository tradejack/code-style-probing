import numpy as np
import torch
import torch.nn as nn
from datasets import load_from_disk, concatenate_datasets
from transformers.data.data_collator import InputDataClass
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from typing import List, Union, Dict
from torch.utils.data.dataloader import DataLoader
from transformers import (
    AutoConfig,
    RobertaTokenizer,
    DefaultDataCollator,
    Seq2SeqTrainer,
    Trainer,
    PreTrainedModel,
    PretrainedConfig,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
import pandas as pd

output_model_name = "combined_nl_prompt_base_features_contd_codet5small"

tokenizer = RobertaTokenizer.from_pretrained(
    "Salesforce/codet5-base", additional_special_tokens=["<nl>", "</nl>"]
)

train_comp_dataset = load_from_disk(
    "datasets/combined_model_nl_prompt/codet5_train_comp_bq_padded.hf"
)
train_casing_dataset = load_from_disk(
    "datasets/combined_model_nl_prompt/codet5_train_case_bq_padded.hf",
).train_test_split(test_size=0.5, seed=42)["train"]
train_docstring_dataset = load_from_disk(
    "datasets/combined_model_nl_prompt/codet5_train_docstring_fixed_bq_padded.hf"
).train_test_split(test_size=0.9, seed=42)["train"]
train_comment_dataset = load_from_disk(
    "datasets/combined_model_nl_prompt/codet5_train_comment_bq_padded.hf"
).train_test_split(test_size=0.5, seed=42)["train"]
train_class_dataset = load_from_disk(
    "datasets/combined_model_nl_prompt/codet5_train_class_bq_padded.hf"
).train_test_split(test_size=0.5, seed=42)["train"]

# test set
test_comp_dataset = load_from_disk(
    "datasets/combined_model_nl_prompt/codet5_test_comp_bq_padded.hf"
)
if len(test_comp_dataset) > 4000:
    p = round(4000 / len(test_comp_dataset), 2)
    test_comp_dataset = test_comp_dataset.train_test_split(
        test_size=p, seed=42
    )["test"]

test_casing_dataset = load_from_disk(
    "datasets/combined_model_nl_prompt/codet5_test_case_bq_padded.hf",
)
if len(test_casing_dataset) > 4000:
    p = round(4000 / len(test_casing_dataset), 2)
    test_casing_dataset = test_casing_dataset.train_test_split(
        test_size=p, seed=42
    )["test"]

test_docstring_dataset = load_from_disk(
    "datasets/combined_model_nl_prompt/codet5_test_docstring_fixed_bq_padded.hf"
)
if len(test_docstring_dataset) > 4000:
    p = round(4000 / len(test_docstring_dataset), 2)
    test_docstring_dataset = test_docstring_dataset.train_test_split(
        test_size=p, seed=42
    )["test"]

test_comment_dataset = load_from_disk(
    "datasets/combined_model_nl_prompt/codet5_test_comment_bq_padded.hf"
)
if len(test_comment_dataset) > 4000:
    p = round(4000 / len(test_comment_dataset), 2)
    test_comment_dataset = test_comment_dataset.train_test_split(
        test_size=p, seed=42
    )["test"]

test_class_dataset = load_from_disk(
    "datasets/combined_model_nl_prompt/codet5_test_class_bq_padded.hf"
)
if len(test_class_dataset) > 4000:
    p = round(4000 / len(test_class_dataset), 2)
    test_class_dataset = test_class_dataset.train_test_split(
        test_size=p, seed=42
    )["test"]


train_codet5_dataset = {
    "comp": train_comp_dataset,
    "case": train_casing_dataset,
    "class": train_class_dataset,
    "docstring": train_docstring_dataset,
    "comment": train_comment_dataset,
}

for dataset in train_codet5_dataset:
    train_codet5_dataset[dataset].set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

test_codet5_dataset = concatenate_datasets(
    [
        test_comp_dataset,
        test_casing_dataset,
        test_docstring_dataset,
        test_comment_dataset,
        test_class_dataset,
    ]
)
test_codet5_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"]
)


class NLPDataCollator:  # (DataCollator):
    """
    Extending the existing DataCollator to work with NLP dataset batches
    """

    def __call__(
        self, features: List[Union[InputDataClass, Dict]]
    ) -> Dict[str, torch.Tensor]:
        # def collate_batch(self, features: List[Union[InputDataClass, Dict]]) -> Dict[str, torch.Tensor]:

        first = features[0]
        if isinstance(first, dict):
            # NLP data sets current works presents features as lists of dictionary
            # (one per example), so we  will adapt the collate_batch logic for that
            if "labels" in first and first["labels"] is not None:
                if np.asarray(first["labels"]).dtype == torch.int64:
                    labels = [f["labels"] for f in features]
                else:
                    labels = [f["labels"] for f in features]
                batch = {"labels": torch.stack(labels)}
            for k, v in first.items():
                if k != "labels" and v is not None and not isinstance(v, str):
                    batch[k] = torch.stack([f[k] for f in features])
            return batch
        else:
            # otherwise, revert to using the default collate_batch
            return DefaultDataCollator().collate_batch(features)


class StrIgnoreDevice(str):
    """
    This is a hack. The Trainer is going call .to(device) on every input
    value, but we need to pass in an additional `task_name` string.
    This prevents it from throwing an error
    """

    def to(self, device):
        return self


class DataLoaderWithTaskname:
    """
    Wrapper around a DataLoader to also yield a task name
    """

    def __init__(self, task_name, data_loader):
        self.task_name = task_name
        self.data_loader = data_loader

        self.batch_size = data_loader.batch_size
        self.dataset = data_loader.dataset

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        for batch in self.data_loader:
            yield batch


class MultitaskDataloader:
    """
    Data loader that combines and samples from multiple single-task
    data loaders.
    """

    def __init__(self, dataloader_dict):
        self.dataloader_dict = dataloader_dict
        self.num_batches_dict = {
            task_name: len(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        self.task_name_list = list(self.dataloader_dict)
        self.dataset = [None] * sum(
            len(dataloader.dataset)
            for dataloader in self.dataloader_dict.values()
        )

    def __len__(self):
        return sum(self.num_batches_dict.values())

    def __iter__(self):
        """
        For each batch, sample a task, and yield a batch from the respective
        task Dataloader.

        We use size-proportional sampling, but you could easily modify this
        to sample from some-other distribution.
        """
        task_choice_list = []
        for i, task_name in enumerate(self.task_name_list):
            task_choice_list += [i] * self.num_batches_dict[task_name]
        task_choice_list = np.array(task_choice_list)
        np.random.shuffle(task_choice_list)
        dataloader_iter_dict = {
            task_name: iter(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        for task_choice in task_choice_list:
            task_name = self.task_name_list[task_choice]
            yield next(dataloader_iter_dict[task_name])


class MultitaskTrainer(Seq2SeqTrainer):
    def get_single_train_dataloader(self, task_name, train_dataset):
        """
        Create a single-task data loader that also yields task names
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        # if is_tpu_available():
        #    train_sampler = get_tpu_sampler(train_dataset)
        # else:
        train_sampler = (
            RandomSampler(train_dataset)
            if self.args.local_rank == -1
            else DistributedSampler(train_dataset)
        )

        data_loader = DataLoaderWithTaskname(
            task_name=task_name,
            data_loader=DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=train_sampler,
                collate_fn=self.data_collator.__call__,  # collate_batch,
            ),
        )

        # if is_tpu_available():
        #    data_loader = pl.ParallelLoader(
        #        data_loader, [self.args.device]
        #    ).per_device_loader(self.args.device)
        return data_loader

    def get_train_dataloader(self):
        """
        Returns a MultitaskDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each 
        task Dataloader
        """
        return MultitaskDataloader(
            {
                task_name: self.get_single_train_dataloader(
                    task_name, task_dataset
                )
                for task_name, task_dataset in self.train_dataset.items()
            }
        )


# train from scratch
# config = AutoConfig.from_pretrained("Salesforce/codet5-small")
# model = AutoModelForSeq2SeqLM.from_config(config)


model = AutoModelForSeq2SeqLM.from_pretrained(
    "seq2seq_results/combined_nl_prompt_base_features_codet5small/epoch 2/checkpoint-85000"
)

batch_size = 16


model.config.max_length = 512
model.resize_token_embeddings(len(tokenizer))
args = Seq2SeqTrainingArguments(
    output_dir=f"seq2seq_results/{output_model_name}",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    # eval_steps=20000,
    eval_accumulation_steps=200,
    learning_rate=2e-3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    # save_total_limit=30,
    num_train_epochs=100,
    predict_with_generate=True,
    push_to_hub=False,
    load_best_model_at_end=True,
)

print(model.num_parameters())

trainer = MultitaskTrainer(
    model,
    args,
    train_dataset=train_codet5_dataset,
    eval_dataset=test_codet5_dataset,
    tokenizer=tokenizer,
    data_collator=NLPDataCollator(),
)

trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))
trainer.train()

eval_dataset = test_codet5_dataset
eval_preds = trainer.predict(eval_dataset)

predictions, labels, report = eval_preds
inputs = test_codet5_dataset["input_ids"]
decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)
decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

df = pd.DataFrame(decoded_preds, columns=["preds"])
df["labels"] = decoded_labels
df["inputs"] = decoded_inputs
df.to_csv(f"seq2seq_results/{output_model_name}/codet5_preds.csv")
