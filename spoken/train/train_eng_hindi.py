import sys
import os
os.environ["HF_HOME"] = "/home/cfiltlab/aakash.agarwal/.cache/huggingface"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

import torch
import argparse
import warnings
import numpy as np
from datasets import load_from_disk, load_dataset, DatasetDict
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from peft import get_peft_model, LoraConfig, TaskType
from IndicTransToolkit import IndicProcessor
from ai4bharat.transliteration import XlitEngine
from spoken.dataset_utils import prepare_dataset

warnings.filterwarnings("ignore", module="huggingface_hub")

import evaluate

bleu_metric = evaluate.load("bleu")


def get_arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="ai4bharat/indictrans2-en-indic-1B")
    parser.add_argument("--output_dir", type=str, default="../models/TED2020_finetuned_hindi_eng")
    parser.add_argument("--data_dir", type=str, default="../datasets/TED2020")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=1e-3)
    parser.add_argument("--metric_for_best_model", type=str, default="eval_loss")
    parser.add_argument("--greater_is_better", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj")
    parser.add_argument("--limit_n", type=int, default=None)


    return parser


def compute_metrics_factory(tokenizer):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        
        # In Seq2Seq, preds can sometimes be a tuple (logits, loss)
        if isinstance(preds, tuple):
            preds = preds[0]

        # Replace -100 (ignored indices) with pad_token_id for decoding
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        decoded_preds = tokenizer.batch_decode(
            preds, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )
        decoded_labels = tokenizer.batch_decode(
            labels, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )

        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        # Printing samples to console during eval as you requested
        print("\n" + "="*21 + " SAMPLE PREDICTIONS " + "="*21)
        for i in range(min(10, len(decoded_preds))):
            print(f"\nExample {i+1}")
            print(f"Prediction: {decoded_preds[i]}")
            print(f"Reference:  {decoded_labels[i]}")
            print("-" * 62)

        # BLEU expects references as a list of lists: [[ref1], [ref2], ...]
        nested_labels = [[l] for l in decoded_labels]
        
        # Calculate metric
        result = bleu_metric.compute(predictions=decoded_preds, references=nested_labels)

        # Return the score (scaled to 100 for readability)
        return {"BLEU": result["bleu"] * 100}

    return compute_metrics



def main(args):

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Running on:", device)

    dataset_dict = prepare_dataset(args.data_dir, limit_n=args.limit_n)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)

    ip = IndicProcessor(inference=False)

    def preprocess_function(examples):

        inputs = examples["english"]
        targets = examples["hindi"]

        processed_inputs = ip.preprocess_batch(
            inputs,
            src_lang="eng_Latn",
            tgt_lang="hin_Deva",
            is_target=False
        )

        processed_targets = ip.preprocess_batch(
            targets,
            src_lang="eng_Latn",
            tgt_lang="hin_Deva",
            is_target=True
        )

        model_inputs = tokenizer(
            processed_inputs,
            max_length=256,
            truncation=True,
            padding=False
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                processed_targets,
                max_length=256,
                truncation=True,
                padding=False
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = dataset_dict.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset_dict["train"].column_names,
    )

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules.split(","),
    )


    model.set_label_smoothing(args.label_smoothing)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size*2,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        predict_with_generate=True,
        generation_max_length=256,
        generation_num_beams=5,
        bf16=torch.cuda.is_available()
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding="longest",
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_factory(tokenizer),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.patience,
                early_stopping_threshold=args.threshold,
            )
        ],
    )

    trainer.train()
    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = get_arg_parse()
    args = parser.parse_args()
    main(args)
