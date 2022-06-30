This data file stores the tokenized versions of data.
To load these datasets use load_from_disk() and to format for the trainer use set_format()

Example code:
train_dataset = load_from_disk('datasets/plbart_sample.hf')
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])