# dataset_utils.py

import os
import ssl
import zipfile
import urllib.request
from datasets import load_from_disk, Dataset, DatasetDict


def limit_splits(dataset, n=100):
    limited = {}
    for split in dataset.keys():
        limited[split] = dataset[split].select(range(min(n, len(dataset[split]))))
    return DatasetDict(limited)


def build_ted2020_dataset(data_dir):
    url = "https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/en-hi.txt.zip"
    zip_path = os.path.join(data_dir, "en-hi.txt.zip")

    os.makedirs(data_dir, exist_ok=True)

    # download if zip not present
    if not os.path.exists(zip_path):
        ssl_context = ssl._create_unverified_context()
        with urllib.request.urlopen(url, context=ssl_context) as response, \
             open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    # extract if not already extracted
    extract_marker = os.path.join(data_dir, "TED2020.en-hi.en")
    if not os.path.exists(extract_marker):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)

    # find .en and .hi files
    en_file = None
    hi_file = None

    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".en"):
                en_file = os.path.join(root, file)
            if file.endswith(".hi"):
                hi_file = os.path.join(root, file)

    if en_file is None or hi_file is None:
        raise FileNotFoundError("Could not locate .en and .hi files.")

    # load corpus
    en_sentences = []
    hi_sentences = []

    with open(en_file, "r", encoding="utf-8") as f_en, \
         open(hi_file, "r", encoding="utf-8") as f_hi:
        for en, hi in zip(f_en, f_hi):
            en_sentences.append(en.strip())
            hi_sentences.append(hi.strip())

    full_dataset = Dataset.from_dict({
        "english": en_sentences,
        "hindi": hi_sentences
    })

    # 70/10/20 split
    split_1 = full_dataset.train_test_split(test_size=0.30, seed=42)
    train_dataset = split_1["train"]
    temp_dataset = split_1["test"]

    split_2 = temp_dataset.train_test_split(test_size=2/3, seed=42)

    dataset = DatasetDict({
        "train": train_dataset,
        "validation": split_2["train"],
        "test": split_2["test"]
    })

    dataset.save_to_disk(data_dir)

    return dataset


def prepare_dataset(dataset_path, limit_n=None):
    try:
        dataset = load_from_disk(dataset_path)
        print("Loaded existing dataset.")
    except Exception:
        print("Building dataset from raw files...")
        dataset = build_ted2020_dataset(dataset_path)
        print("Saved dataset.")

    if limit_n is not None:
        dataset = limit_splits(dataset, n=limit_n)

    return dataset
