import sys
import os
os.environ["HF_HOME"] = "/home/cfiltlab/aakash.agarwal/.cache/huggingface"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

import torch
import evaluate
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor
from peft import PeftModel
import warnings
from spoken.dataset_utils import prepare_dataset
from tqdm.auto import tqdm
from math import ceil

warnings.filterwarnings("ignore", module="huggingface_hub")


def get_arg_parse():
    parser = argparse.ArgumentParser(description="Evaluation script for HI to EN translation")

    parser.add_argument("--data_dir", type=str, default="../datasets/TED2020")
    parser.add_argument("--base_ckpt", type=str, default="ai4bharat/indictrans2-indic-en-1B")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="../outputs")
    def none_or_int(value):
        if value == "None":
            return None
        return int(value)

    parser.add_argument("--limit_n", type=none_or_int, default=None)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--use_peft", action="store_true")

    return parser.parse_args()


def translate_batch(model, tokenizer, sentences, device, batch_size=8):

    ip = IndicProcessor(inference=True)
    translations = []
    model.eval()

    num_batches = ceil(len(sentences) / batch_size)

    for i in tqdm(
        range(0, len(sentences), batch_size),
        total=num_batches,
        desc="Translating",
        unit="batch"
    ):

        batch = sentences[i:i + batch_size]
        batch = ip.preprocess_batch(
            batch,
            src_lang="hin_Deva",
            tgt_lang="eng_Latn"
        )

        inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True
        ).to(device)

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                max_length=256,
                num_beams=2,
                num_return_sequences=1
            )

        decoded_preds = tokenizer.batch_decode(
            generated_tokens.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        postprocessed_preds = ip.postprocess_batch(
            decoded_preds,
            lang="eng_Latn"
        )

        translations.extend(postprocessed_preds)

    return translations




def compute_metrics(predictions, references, sources, batch_size=64, num_workers=max((os.cpu_count() or 1) - 2, 1)):

    results = {}

    # BLEU
    metric_bleu = evaluate.load("bleu")
    results["bleu"] = metric_bleu.compute(
        predictions=predictions,
        references=references
    )["bleu"]
    print(f"BLEU: {results['bleu']}")

    # CHRF
    metric_chrf = evaluate.load("chrf")
    results["chrf"] = metric_chrf.compute(
        predictions=predictions,
        references=references
    )["score"]
    print(f"CHRF: {results['chrf']}")

    # BERTScore
    metric_bertscore = evaluate.load("bertscore")
    bert_score = metric_bertscore.compute(
        predictions=predictions,
        references=references,
        lang="en"
    )
    results["bertscore_f1"] = sum(bert_score["f1"]) / len(bert_score["f1"])
    print(f"BERTScore F1: {results['bertscore_f1']}")

    # COMET
    try:
        from comet import download_model, load_from_checkpoint

        model_path = download_model("Unbabel/wmt22-comet-da")
        comet_model = load_from_checkpoint(model_path)

        data = [
            {"src": s, "mt": m, "ref": r}
            for s, m, r in zip(sources, predictions, references)
        ]

        gpus = 1
        model_output = comet_model.predict(
            data,
            batch_size=batch_size,
            gpus=gpus
        )

        results["comet"] = model_output.system_score
        print(f"COMET: {results['comet']}")

    except Exception as e:
        results["comet"] = f"Error: {e}"
        print(f"COMET: {results['comet']}")

    # BLEURT
    try:
        metric_bleurt = evaluate.load(
            "bleurt",
            config_name="BLEURT-20",
        )

        bleurt_score = metric_bleurt.compute(
            predictions=predictions,
            references=references,
            batch_size=batch_size
        )

        results["bleurt"] = sum(bleurt_score["scores"]) / len(bleurt_score["scores"])
        print(f"BLEURT: {results['bleurt']}")

    except Exception as e:
        results["bleurt"] = f"Error: {e}"
        print(f"BLEURT: {results['bleurt']}")

    return results




def main():

    args = get_arg_parse()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Running on:", device)

    dataset_dict = prepare_dataset(args.data_dir, limit_n=args.limit_n)
    test_data = dataset_dict["test"]

    english_sentences = test_data["english"]
    hindi_sentences = test_data["hindi"]

    os.makedirs(args.output_dir, exist_ok=True)

    output_file = (
        "hindi_to_en_results_finetuned.csv"
        if args.use_peft
        else "hindi_to_en_results_base.csv"
    )

    output_path = os.path.join(args.output_dir, output_file)

    if os.path.exists(output_path):

        print("Output file exists. Loading predictions for evaluation...")
        df = pd.read_csv(output_path)

        preds_en = df["english_pred"].tolist()
        english_sentences = df["english"].tolist()
        hindi_sentences = df["hindi"].tolist()

    else:

        tokenizer = AutoTokenizer.from_pretrained(
            args.base_ckpt,
            trust_remote_code=True
        )

        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            args.base_ckpt,
            trust_remote_code=True,
            torch_dtype=torch.float32
        ).to(device)

        model = base_model

        if args.use_peft:
            if args.adapter_path is None:
                raise ValueError("adapter_path must be provided when --use_peft is set")

            print("Loading fine-tuned PEFT model...")
            model = PeftModel.from_pretrained(base_model, args.adapter_path)
            model = model.merge_and_unload()
        else:
            print("Using base model only...")

        model.to(device)
        model.eval()

        print("Generating predictions...")

        preds_en = translate_batch(
            model,
            tokenizer,
            hindi_sentences,
            device,
            batch_size=args.batch_size
        )

        df = pd.DataFrame({
            "hindi": hindi_sentences,
            "english": english_sentences,
            "english_pred": preds_en
        })

        df.to_csv(output_path, index=False)
        print("Saved predictions to:", output_path)

    metrics = compute_metrics(
        preds_en,
        english_sentences,
        hindi_sentences,
        batch_size=args.batch_size
    )

    print("Final Metrics:", metrics)


if __name__ == "__main__":
    main()
