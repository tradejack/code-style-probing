# Code Style Transfer & Probing
UCSC IBM Capstone dedicated to probing large language models for code style.


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
```bash
python combined_parallel_gen_script.py FEATURES
```

**FEATURES**: Any combination of the feature that to be transferred and should be combined with `+`, i.e. comment+docstring

### Parallel Corpus Tokenization

Tokenizing and filtering out all the NULL value examples.
```bash
python parallel_preprocessing_script.py FEATURES
```

**FEATURES**: Any combination of the feature that to be transferred and should be combined with `+`, i.e. casing+class+list_comp+comment+docstring

> Preprocessing on the docstring transfer will be done for removing very long sequence data.


### Splitting Long Sequences
For keeping dataset in the range of the max sequence length, use this script for filtering out the short sequence data and extracting function/class level codes out of the long sequences. The output will be 2 files: short and long datasets. 
```bash
python split_long_data.py [train|eval]
```

> You will need to configure the file path inside the script.

### Adding Control Tokens
```bash
python add_control_code_script.py \
    INPUT_DATASET_PATH \
    OUTPUT_DATASET_PATH \
    FEATURES \
    CONTROL_TOKEN_TYPE
	datasets/seq2seq_datasets/codet5_train_comp_bq_padded.hf \
	datasets/combined_model_nl_prompt/codet5_train_comp_bq_padded.hf \
	list_comp \
	2
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
    
## Model Training and Inference
### GAN Training
> Modify the setup in `config.py` before starting the training.
> This may not work really well, need to make deeper inspection on the training process.
> The `gpu.py` will select the free-most GPU for training.
```bash
export CUDA_VISIBLE_DEVICES=$(python gpu.py | tail -n 1); python train.py
```

### Individual Finetuning for Seq2Seq Model
The Seq2Seq Generation finetuning with CodeT5 on individual feature transfer. 
```bash
export CUDA_VISIBLE_DEVICES=$(python gpu.py | tail -n 1); python seq2seq_train.py
```
> You will need to configure the training data in the script

### Combined Finetuning for Seq2Seq Model
The Seq2Seq Generation finetuning with CodeT5 on combined feature transfer. 

```bash
export CUDA_VISIBLE_DEVICES=$(python gpu.py | tail -n 1); python combined_seq2seq_train.py
```
> You will need to configure the training data in the script

### Seq2Seq Inference
```bash
export CUDA_VISIBLE_DEVICES=$(python gpu.py | tail -n 1); \
    python seq2seq_inference.py \
    DATASET_PATH \
    CHECKPOINT \
    OUTPUT_FILE_PATH \
    [IS_NL] \
    [IS_DOWNSIZE]
```

**DATASET_PATH**: The path of the test set. (`.hf`)
**CHECKPOINT**: The model checkpoint path.
**OUTPUT_FILE_PATH**: The path of the prediction output
**IS_NL**: [true|false], whether use the control tokens.
**IS_DOWNSIZE**: [true|false], whether need to downsize the test set, will downsize it to 2000 examples.
