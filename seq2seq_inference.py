from pathlib import Path

import typer


def main(
    inference_dataset,
    model_ckpt_path,
    output_csv_filename,
    batch_size: int = 8,
    is_nl: bool = False,
    is_downsize: bool = False,
):

    import pandas as pd
    from datasets import load_from_disk
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
    )

    output_dir = "/".join(output_csv_filename.split("/")[:-1])
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    test_codet5_dataset = load_from_disk(inference_dataset)

    tokenizer = AutoTokenizer.from_pretrained(model_ckpt_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt_path)
    model.config.max_length = 512
    if is_nl:
        model.resize_token_embeddings(len(tokenizer))

    args = Seq2SeqTrainingArguments(
        output_dir=f"{model_ckpt_path}",
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
        train_dataset=test_codet5_dataset,
        eval_dataset=test_codet5_dataset,
        # data_collator=data_collator,
        tokenizer=tokenizer,
    )
    if is_downsize and len(test_codet5_dataset) > 2000:
        p = round(2000 / len(test_codet5_dataset), 4)
        test_codet5_dataset = test_codet5_dataset.train_test_split(
            test_size=p, seed=1234
        )["test"]

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
    df.to_csv(f"{output_csv_filename}")


if __name__ == "__main__":
    typer.run(main)
