import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_from_disk
from multiprocessing import Pool, cpu_count


DATASET_ROOT = "datasets_final"
STATS_ROOT = os.path.join(DATASET_ROOT, "stats")
PLOTS_ROOT = os.path.join(STATS_ROOT, "plots")


def process_language(args):
    dataset_split, language = args

    word_counts = []
    char_lengths = []

    for example in dataset_split:
        text = example.get(language, None)
        if not isinstance(text, str):
            continue

        text = text.strip()
        words = text.split()

        word_counts.append(len(words))
        char_lengths.append(len(text))

    word_counts = np.array(word_counts)
    char_lengths = np.array(char_lengths)

    return language, {
        "num_sentences": int(len(word_counts)),
        "total_words": int(word_counts.sum()),
        "word_mean": float(word_counts.mean()) if len(word_counts) else 0.0,
        "word_std": float(word_counts.std()) if len(word_counts) else 0.0,
        "word_min": int(word_counts.min()) if len(word_counts) else 0,
        "word_max": int(word_counts.max()) if len(word_counts) else 0,
        "char_mean": float(char_lengths.mean()) if len(char_lengths) else 0.0,
        "char_std": float(char_lengths.std()) if len(char_lengths) else 0.0,
    }


def compute_split_statistics(dataset_split, split_name):
    print(f"Processing split: {split_name}")
    languages = list(dataset_split.features.keys())

    with Pool(processes=min(len(languages), cpu_count())) as pool:
        results = pool.map(
            process_language,
            [(dataset_split, lang) for lang in languages]
        )

    return dict(results)


def save_dataset_json(dataset_name, all_split_stats):
    os.makedirs(STATS_ROOT, exist_ok=True)
    output_file = os.path.join(STATS_ROOT, f"{dataset_name}_stats.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_split_stats, f, indent=4, ensure_ascii=False)


def plot_combined_stats(aggregated_stats):
    os.makedirs(PLOTS_ROOT, exist_ok=True)

    # Load all JSON stats
    dataset_files = [
        f for f in os.listdir(STATS_ROOT)
        if f.endswith("_stats.json")
    ]

    all_data = {}

    for file in dataset_files:
        dataset_name = file.replace("_stats.json", "")
        with open(os.path.join(STATS_ROOT, file), "r", encoding="utf-8") as f:
            all_data[dataset_name] = json.load(f)

    splits = ["train"]

    for split in splits:
        datasets = []
        avg_word_means = []
        avg_char_means = []
        avg_word_stds = []
        total_words = []
        total_sentences = []
        eng_means = []
        hin_means = []

        for dataset_name, stats in all_data.items():
            if split not in stats:
                continue

            split_stats = stats[split]

            word_means = []
            char_means = []
            word_stds = []
            totals = []
            sentence_counts = []

            for lang, values in split_stats.items():
                word_means.append(values["word_mean"])
                char_means.append(values["char_mean"])
                word_stds.append(values["word_std"])
                totals.append(values["total_words"])
                sentence_counts.append(values["num_sentences"])

                if lang == "english":
                    eng_means.append(values["word_mean"])
                if lang == "hinglish":
                    hin_means.append(values["word_mean"])

            datasets.append(dataset_name)
            avg_word_means.append(np.mean(word_means))
            avg_char_means.append(np.mean(char_means))
            avg_word_stds.append(np.mean(word_stds))
            total_words.append(np.sum(totals))
            total_sentences.append(np.sum(sentence_counts))

        if not datasets:
            continue

        # 1 Average word count
        plt.figure()
        plt.bar(datasets, avg_word_means)
        plt.xticks(rotation=45)
        plt.title(f"{split} - Average Word Count")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_ROOT, f"{split}_avg_wordcount.png"))
        plt.close()

        # 2 Average character length
        plt.figure()
        plt.bar(datasets, avg_char_means)
        plt.xticks(rotation=45)
        plt.title(f"{split} - Average Character Length")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_ROOT, f"{split}_avg_charlength.png"))
        plt.close()

        # 3 Word count standard deviation
        plt.figure()
        plt.bar(datasets, avg_word_stds)
        plt.xticks(rotation=45)
        plt.title(f"{split} - Word Count Standard Deviation")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_ROOT, f"{split}_word_std.png"))
        plt.close()

        # 4 Total words
        plt.figure()
        plt.bar(datasets, total_words)
        plt.xticks(rotation=45)
        plt.title(f"{split} - Total Words")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_ROOT, f"{split}_total_words.png"))
        plt.close()

        # 5 Total sentence count (NEW)
        plt.figure()
        plt.bar(datasets, total_sentences)
        plt.xticks(rotation=45)
        plt.title(f"{split} - Total Sentence Count")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_ROOT, f"{split}_total_sentences.png"))
        plt.close()

        # 6 English vs Hindi mean comparison
        if len(eng_means) == len(hin_means) and len(eng_means) > 0:
            x = np.arange(len(datasets))
            width = 0.35

            plt.figure()
            plt.bar(x - width / 2, eng_means, width, label="English")
            plt.bar(x + width / 2, hin_means, width, label="Hinglish")
            plt.xticks(x, datasets, rotation=45)
            plt.legend()
            plt.title(f"{split} - English vs Hinglish Mean Word Count")
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_ROOT, f"{split}_eng_vs_hin.png"))
            plt.close()

        # 7 Boxplot of mean word counts
        plt.figure()
        plt.boxplot(avg_word_means)
        plt.title(f"{split} - Distribution of Avg Word Counts Across Datasets")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_ROOT, f"{split}_boxplot_wordmeans.png"))
        plt.close()

        # 8 Histogram of average word counts
        plt.figure()
        plt.hist(avg_word_means, bins=10)
        plt.title(f"{split} - Histogram of Avg Word Counts")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_ROOT, f"{split}_hist_wordmeans.png"))
        plt.close()

    print("All plots generated.")

def main():
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    aggregated_stats = {
        "train": {},
        "validation": {},
        "test": {}
    }

    for dataset_name in os.listdir(DATASET_ROOT):
        dataset_path = os.path.join(DATASET_ROOT, dataset_name)

        if dataset_name == "stats":
            continue

        if not os.path.isdir(dataset_path):
            continue

        output_file = os.path.join(STATS_ROOT, f"{dataset_name}_stats.json")

        if os.path.exists(output_file):
            print(f"Skipping {dataset_name}, JSON already exists.")
            continue

        print(f"\nLoading dataset: {dataset_name}")
        dataset = load_from_disk(dataset_path)

        all_split_stats = {}

        for split in dataset.keys():
            split_stats = compute_split_statistics(dataset[split], split)
            all_split_stats[split] = split_stats

            avg_word_mean = np.mean([
                lang_stats["word_mean"]
                for lang_stats in split_stats.values()
            ])

            if split in aggregated_stats:
                aggregated_stats[split][dataset_name] = float(avg_word_mean)

        save_dataset_json(dataset_name, all_split_stats)
        print(f"Saved JSON for {dataset_name}")

    plot_combined_stats(aggregated_stats)
    print("All done.")


if __name__ == "__main__":
    main()
