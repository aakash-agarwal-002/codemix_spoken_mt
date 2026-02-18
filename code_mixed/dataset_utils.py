# dataset_utils.py

import os
from datasets import load_from_disk, load_dataset, DatasetDict
from ai4bharat.transliteration import XlitEngine


def limit_splits(dataset, n=100):
    limited = {}
    for split in dataset.keys():
        limited[split] = dataset[split].select(range(min(n, len(dataset[split]))))
    return DatasetDict(limited)


def add_transliteration_column(dataset):
    engine = XlitEngine("hi", beam_width=5)

    def transliterate_batch(batch):
        transliterated = []
        for s in batch["hinglish"]:
            if s is None:
                transliterated.append("")
            else:
                result = engine.translit_sentence(s)
                if isinstance(result, dict):
                    transliterated.append(result.get("hi", ""))
                else:
                    transliterated.append(result)

        return {"hinglish_dev": transliterated}

    for split in dataset.keys():
        if "hinglish_dev" not in dataset[split].column_names:
            dataset[split] = dataset[split].map(
                transliterate_batch,
                batched=True,
                batch_size=32,
                num_proc=os.cpu_count(),
            )

    return dataset


def prepare_dataset(dataset_path, limit_n=None):
    dataset_name = "LingoIITGN/PHINC"
    hf_marker = os.path.join(dataset_path, "dataset_dict.json")

    if os.path.exists(hf_marker):
        dataset = load_from_disk(dataset_path)
        if "hinglish_dev" not in dataset["train"].column_names:
            dataset = add_transliteration_column(dataset)
            dataset.save_to_disk(dataset_path)
    else:
        os.makedirs(dataset_path, exist_ok=True)
        dataset = load_dataset(dataset_name)

        if "train" in dataset and len(dataset) == 1:
            full_dataset = dataset["train"]
            split_1 = full_dataset.train_test_split(test_size=0.30, seed=42)
            temp_dataset = split_1["test"]
            split_2 = temp_dataset.train_test_split(test_size=2/3, seed=42)

            dataset = DatasetDict({
                "train": split_1["train"],
                "validation": split_2["train"],
                "test": split_2["test"]
            })

        for split in dataset.keys():
            if "Sentence" in dataset[split].column_names:
                dataset[split] = dataset[split].rename_columns({
                    "Sentence": "hinglish",
                    "English_Translation": "english"
                })

        dataset = add_transliteration_column(dataset)
        dataset.save_to_disk(dataset_path)

    if limit_n is not None:
        dataset = limit_splits(dataset, n=limit_n)

    return dataset
