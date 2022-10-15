# Code Style Transfer & Probing
UCSC IBM Capstone dedicated to probing large language models for code style.
- [Code Style Transfer & Probing](#code-style-transfer---probing)
  * [Data Preprocessing](#data-preprocessing)
    + [Extract Metrics from Py150k](#extract-metrics-from-py150k)
    + [Raw Script Tokenization](#raw-script-tokenization)
    + [Individual Feature Parallel Corpora Generation](#individual-feature-parallel-corpora-generation)
    + [Combined Feature Parallel Corpora Generation](#combined-feature-parallel-corpora-generation)
    + [Splitting Long Sequences](#splitting-long-sequences)
  * [GAN Training](#gan-training)
      - [GAN Training(not used)](#gan-training-not-used-)
  * [End-to-End Training Pipeline Instruction](#end-to-end-training-pipeline-instruction)
    + [1. Parallel Corpus Tokenization](#1-parallel-corpus-tokenization)
      - [Example](#example)
    + [2. Add Control Tokens(combined model only)](#2-add-control-tokens-combined-model-only-)
      - [Example](#example-1)
    + [3. Training](#3-training)
      - [Finetuning for Seq2Seq Model](#finetuning-for-seq2seq-model)
    + [4. Seq2Seq Inference](#4-seq2seq-inference)
    + [5. Evaluation](#5-evaluation)

## Data Preprocessing
### Extract Metrics from Py150k
```bash
python extract_metrics.py py150k
```
### Raw Script Tokenization

The script is used for directly tokenizing the full dataset
- `train`: training set that named `bq_data_outlier.csv`
- `eval`: eval set that named `evaluation_set.csv`
```bash
python tokenize_raw_script.py [train|eval]
```

### Individual Feature Parallel Corpora Generation
Generate all individual features of parallel corpora at once.
```bash
python parallel_corpora_gen_script.py \
    INPUT_CSV_PATH \
    OUTPUT_CSV_PATH
```
**INPUT_CSV_PATH**: the path for the input csv data, which will contain the raw script. The `evaluation_set.csv` gives the correct column names.
**OUTPUT_CSV_PATH**: the path for the output file. It will be a csv file containing all the script that the individual features are transferred.

### Combined Feature Parallel Corpora Generation
```
python combined_parallel_gen_script.py FEATURES
Example: 
python combined_parallel_gen_script.py comment+docstring \
--csv-name /data/curated_eval_set/eval_set_short_individual_feat.csv \
--output-dir /data/curated_eval_set \
--is-short True

Arguments:
  TARGET_FEAT  [required]

Options:
  --csv-name TEXT
  --output-dir TEXT
  --is-short TEXT                 [default: False]
```

- **TARGET_FEAT**: Any combination of the feature that to be transferred and should be combined with `+`, i.e. comment+docstring

- `csv-name`: Input CSV file, should be the file with all individual feature transfer scripts.

- `output-dir`: The output location

- `is-short`: Whether the input script is the shorten version(in the range of the max length)



### Splitting Long Sequences
For keeping dataset in the range of the max sequence length, use this script for filtering out the short sequence data and extracting function/class level codes out of the long sequences. The output will be 2 files: short and long datasets. 
```bash
python split_long_data.py [train|eval]
```

> You will need to configure the file path inside the script.
## GAN Training
#### GAN Training(not used)
> Modify the setup in `config.py` before starting the training.
> This may not work really well, need to make deeper inspection on the training process.
> The `gpu.py` will select the free-most GPU for training.
```bash
export CUDA_VISIBLE_DEVICES=$(python gpu.py | tail -n 1); python train.py
```

## End-to-End Training Pipeline Instruction
### 1. Parallel Corpus Tokenization

Tokenizing and filtering out all the NULL value examples.
```bash
python parallel_preprocessing_script.py FEATURES CSV_NAME OUTPUT_PATH
```

**FEATURES**: Any combination of the feature that to be transferred and should be combined with `+`, i.e. casing+class+list_comp+comment+docstring

**CSV_NAME**: CSV file that contains all individual features
- **train set - casing:** `bq_data_uncased.csv`
- **train set - class:** `bq_data_outlier_no_class.csv`
- **train set - list comp:** `bq_data_uncomp_fixed_outlier.csv`
- **train set - comment:** `bq_uncommented_outlier.csv`
- **train set - docstring:** `bq_updated_docstring_outlier.csv`
- **eval set:** `eval_set_individual_feat.csv`
    - eval set contains separate labels for each individual transformation

**OUTPUT_PATH**: The output dataset path, whatever you want, which will be a `.hf` file

> Preprocessing on the docstring transfer will be done for removing very long sequence data.
#### Example
```bash
# inidividual
## i.e. class
### train
python parallel_preprocessing_script.py \
    class \
    bq_data_outlier_no_class.csv \
    train_class_dataset.hf
### eval
python parallel_preprocessing_script.py \
    class \
    eval_set_individual_feat.csv \
    eval_class_dataset.hf

# combined - eval only
## i.e. class+list_comp
python parallel_preprocessing_script.py \
    class+list_comp \
    eval_set_individual_feat.csv \
    eval_class_list_comp_dataset.hf
```
### 2. Add Control Tokens(combined model only)
Input: tokenized `.hf` file.
Output: tokenized `.hf` file containing control tokens.
You can use this to finetune or make prediction with combined model.
```bash
python add_control_code_script.py \
    INPUT_DATASET_PATH \
    OUTPUT_DATASET_PATH \
    FEATURES \
    CONTROL_TOKEN_TYPE
```
**INPUT_DATASET_PATH**: the input dataset file path, should be `.hf` file.
**OUTPUT_DATASET_PATH**: the output dataset file path, should be `.hf` file.
**FEATURES**: Any combination of the feature that to be transferred and should be combined with `+`, i.e. casing+class+list_comp+comment+docstring.
**CONTROL_TOKEN_TYPE**: there are 4 types of control tokens
1. `# (comp|case|comment|docstring|class)` and appended at the end of the sequence.
2. Use natural language for describing the transfer such as "change for loop to list comprehension" and wrap it with `<nl>prompt</nl>`
3. Same as **2**, but considering `<nl>` and `</nl>` as special tokens(new tokens to train and **2** was not).
4. Same as **2**, but simplifying the prompt sentences.
> 4 performed the best.
#### Example
```bash
# individual features (finetune only)
## i.e. class
python add_control_code_script.py \
    train_class_dataset.hf \
    train_class_dataset_with_tokens.hf \
    class \
    4

# multiple features (inference only)
## i.e. class+list comp
python add_control_code_script.py \
    eval_class_list_comp_dataset.hf \
    eval_class_list_comp_dataset_with_tokens.hf \
    class+list_comp \
    4
```

### 3. Training
#### Finetuning for Seq2Seq Model
The Seq2Seq Generation finetuning with CodeT5. 
```bash
# individual
export CUDA_VISIBLE_DEVICES=$(python gpu.py | tail -n 1); python seq2seq_train.py
# combined
export CUDA_VISIBLE_DEVICES=$(python gpu.py | tail -n 1); python combined_seq2seq_train.py
```
You will need to configure the training in the script:
**seq2seq_train.py**
1. `fname_prefix`: your repo directory i.e. `/home/you/code-style-probing/`
2. `train_dataset_hf_name`: train set. But in the script, we dowsized it due to the training time constraint. i.e. `train_class_dataset.hf`
3. `test_dataset_hf_name`: test set. i.e. `test_class_dataset.hf`
4. `output_dir_name`: checkpoint folder  i.e.`codet5-class-checkpoints/`
5. `model_checkpoint`: checkpoint name, can be the folder or huggingface checkpoint, i.e. `Salesforce/codet5-small`
6. `inference_only`: whether only do the inference on the test set,  i.e. `False`
7. `down_size_test_set`: whether downsize the test set for saving time. i.e. `True`
8. `is_baseline`: if baseline, the CodeT5 will be trained from scratch.  i.e. `False`
9. `batch_size`: i.e. `16`

**combined_seq2seq_train.py**
1. `batch_size`: `16`
2. `output_model_name`: output model folder name, i.e. `combined_nl_prompt_base_features_contd_codet5small`
3. `checkpoint`:checkpoint directory, i.e. `seq2seq_results/combined_nl_prompt_base_features_contd_codet5small/epoch 2/checkpoint-85000`
4. `train_comp_dataset`: train set, control tokens need to be added.
4. `train_casing_dataset`: train set, control tokens need to be added.
4. `train_docstring_dataset`: train set, control tokens need to be added.
4. `train_comment_dataset`: train set, control tokens need to be added.
4. `train_class_dataset`: train set, control tokens need to be added.
4. `test_comp_dataset`: another split for validation, control tokens need to be added.
4. `test_casing_dataset`: another split for validation, control tokens need to be added.
4. `test_docstring_dataset`: another split for validation, control tokens need to be added.
4. `test_comment_dataset`: another split for validation, control tokens need to be added.
4. `test_class_dataset`: another split for validation, control tokens need to be added.


### 4. Seq2Seq Inference
```
Usage: seq2seq_inference.py [OPTIONS] INFERENCE_DATASET MODEL_CKPT_PATH
                            OUTPUT_CSV_FILENAME

Arguments:
  INFERENCE_DATASET    [required]
  MODEL_CKPT_PATH      [required]
  OUTPUT_CSV_FILENAME  [required]

Options:
  --batch-size INTEGER            [default: 8]
  --is-nl / --no-is-nl            [default: no-is-nl]
  --is-downsize / --no-is-downsize
                                  [default: no-is-downsize]

Example:
rm -rf codestylist ; \
export CUDA_VISIBLE_DEVICES=1; \
python seq2seq_inference.py \
/data/code/curated_eval_set/curated_docstring_dataset_with_prompt.hf \
codestylist/combined_code_style_transformer \
combined_model_results/docstring.non_downsized.output.csv \
--batch-size 64 \
--is-nl ;
```

- **DATASET_PATH**: The path of the test set. (`.hf`)
- **CHECKPOINT**: The model checkpoint path.
- **OUTPUT_FILE_PATH**: The path of the prediction output
- **IS_NL**: [true|false], whether use the control tokens.
- **IS_DOWNSIZE**: [true|false], whether need to downsize the test set, will downsize it to 2000 examples.

The output will be a prediction file that contains input/prediction/label.
> The removal of `codestylist` folder is because the trainer will create a foler automatically and will have error if we try to load the model from the hub, it will try to load from the empty folder created by trainer instead. So it is needed to remove the folder first no matter whether it exists.
### 5. Evaluation
Please see `seq2seq_eval.ipynb`(individual) and `combined_seq2seq_eval.ipynb`(combined) for evaluation.

#### Script Usage
We now have a script `evaluate_score` for running the evaluation:
```
Usage: evaluate_score.py [OPTIONS] PRED_DIR OUTPUT_DIR TARGET_FEAT

Arguments:
  PRED_DIR     [required]
  OUTPUT_DIR   [required]
  TARGET_FEAT  [required]

Options:
  --is-nl-tokens-added / --no-is-nl-tokens-added
                                  [default: no-is-nl-tokens-added]
  --clean-diff / --no-clean-diff  [default: clean-diff]

Example:
python evaluate_score.py \
/data/ken/data/code/decorator.output_post_process.csv \
./test.json decorator \
--clean-diff
```

- PRED_DIR: You prediction csv file
- OUTPUT_DIR: You score output json file name
- is-nl-tokens-added: if true, will run preprocessing on removing nl prompt(combined model only)
- clean-diff: will clean some inconsistent characters caused by AST parse and unparse before calculating DiffBLEU
