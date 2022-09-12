# Constants
MY_PATH = "/data/users/cting3/CodeNest/code-style-probing/"
SHARED_PATH = "/data/users/team2_capstone/code-style-probing/"

COMBINED_DATASET_TRAIN = (
    "datasets/combined_model/codet5_train_combined_base.hf"
)
COMBINED_DATASET_TEST = "datasets/combined_model/codet5_test_combined_base.hf"
COMBINED_DIR = "combined_base_features_codet5small"

NO_CLASS_NO_SUPER_TRAIN = (
    "datasets/seq2seq_datasets/codet5_train_no_class_no_super_bq_padded.hf"
)
NO_CLASS_NO_SUPER_TEST = (
    "datasets/seq2seq_datasets/codet5_test_no_class_no_super_bq_padded.hf"
)
NO_CLASS_NO_SUPER_DIR = "outlier_no_class_no_super_codet5small"

# Config
fname_prefix = MY_PATH
train_dataset_hf_name = COMBINED_DATASET_TRAIN
test_dataset_hf_name = COMBINED_DATASET_TEST
output_dir_name = COMBINED_DIR
# model_checkpoint = "seq2seq_results/outlier_docstring_codet5small/checkpoint-46500"  # "Salesforce/codet5-small"
model_checkpoint = "Salesforce/codet5-small"
inference_only = False
down_size_test_set = True
is_baseline = True

batch_size = 16


import pandas as pd

from transformers import *
from datasets import load_from_disk

train_codet5_dataset = load_from_disk(
    fname_prefix + f"{train_dataset_hf_name}"
).train_test_split(test_size=0.2)
test_codet5_dataset = load_from_disk(fname_prefix + f"{test_dataset_hf_name}")

if is_baseline:
    # train from scratch
    config = T5Config.from_pretrained("Salesforce/codet5-small")
    model = AutoModelForSeq2SeqLM.from_config(config)
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# eval_dataset = test_codet5_dataset
# eval_dataset = eval_dataset.train_test_split(test_size = 0.08)
# print (eval_dataset)

model_name = model_checkpoint.split("/")[-1]
from transformers import AutoConfig

# config = AutoConfig.from_pretrained(fname_prefix + "/seq2seq_results/no_outlier_codet5small/checkpoint-51000",    min_length = 20, max_length = 512, )
# model =AutoModelForSeq2SeqLM.from_pretrained(fname_prefix + "/seq2seq_results/outlier_codet5small/checkpoint-40500")
# model =AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-small")
# model =AutoModelForSeq2SeqLM.from_pretrained(fname_prefix + "/seq2seq_results/outlier_docstring_codet5small/checkpoint-46000")

model.config.max_length = 512
# fname_prefix + /seq2seq_results/no_outlier_codet5small/checkpoint-50500
args = Seq2SeqTrainingArguments(
    # f"{model_name}-finetuned-xsum",
    output_dir=fname_prefix + f"/seq2seq_results/{output_dir_name}",
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
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=train_codet5_dataset["train"],
    eval_dataset=train_codet5_dataset["test"],
    # data_collator=data_collator,
    tokenizer=tokenizer,
)
trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))
# model.generate()
#
if not inference_only:
    trainer.train()
# data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# downsize the test set
if down_size_test_set:
    test_codet5_dataset = test_codet5_dataset.train_test_split(test_size=0.01)
    test_codet5_dataset = test_codet5_dataset["test"]

eval_preds = trainer.predict(
    test_codet5_dataset
)  # , min_length=40, max_length = 512)

predictions, labels, report = eval_preds
inputs = test_codet5_dataset["input_ids"]
decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)
decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

df = pd.DataFrame(decoded_preds, columns=["preds"])
df["labels"] = decoded_labels
df["inputs"] = decoded_inputs
# print (decoded_labels[0])
df.to_csv(fname_prefix + f"seq2seq_results/{output_dir_name}/codet5_preds.csv")
