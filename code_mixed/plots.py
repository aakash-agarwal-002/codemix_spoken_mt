import os
import json
import re
import matplotlib.pyplot as plt

base_dirs = [
    "/home/cfiltlab/aakash.agarwal/ai4bharat/code_mixed/models/phinc_finetuned_eng_hinglish",
    "/home/cfiltlab/aakash.agarwal/ai4bharat/code_mixed/models/phinc_finetuned_hinglish_eng",
]

output_root = "plots"


def find_highest_checkpoint(base_path):
    checkpoint_dirs = []

    for name in os.listdir(base_path):
        match = re.match(r"checkpoint-(\d+)", name)
        if match:
            step = int(match.group(1))
            checkpoint_dirs.append((step, name))

    if not checkpoint_dirs:
        return None

    checkpoint_dirs.sort(key=lambda x: x[0])
    return os.path.join(base_path, checkpoint_dirs[-1][1])


def extract_eval_metrics(trainer_state_path):
    with open(trainer_state_path, "r") as f:
        trainer_state = json.load(f)

    log_history = trainer_state.get("log_history", [])

    epochs = []
    eval_bleu = []
    eval_loss = []

    for entry in log_history:
        if "eval_BLEU" in entry:
            epochs.append(entry["epoch"])
            eval_bleu.append(entry["eval_BLEU"])
            eval_loss.append(entry["eval_loss"])

    return epochs, eval_bleu, eval_loss



for base_dir in base_dirs:

    base_dir_name = os.path.basename(os.path.normpath(base_dir))
    print(f"Processing {base_dir_name}")

    highest_checkpoint_path = find_highest_checkpoint(base_dir)

    if highest_checkpoint_path is None:
        print("  No checkpoints found")
        continue

    trainer_state_path = os.path.join(highest_checkpoint_path, "trainer_state.json")

    if not os.path.exists(trainer_state_path):
        print("  trainer_state.json not found")
        continue

    epochs, eval_bleu, eval_loss = extract_eval_metrics(trainer_state_path)

    if len(epochs) == 0:
        print("  No evaluation metrics found")
        continue

    output_dir = os.path.join(output_root, base_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    pdf_output_path = os.path.join(output_dir, "loss_bleu_plots.pdf")

    # Create 1x2 plot
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, eval_bleu)
    plt.xlabel("Epoch")
    plt.ylabel("eval_BLEU")
    plt.title("eval_BLEU vs Epoch")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, eval_loss)
    plt.xlabel("Epoch")
    plt.ylabel("eval_loss")
    plt.title("eval_loss vs Epoch")

    plt.tight_layout()

    # Save directly as PDF
    plt.savefig(pdf_output_path, format="pdf")
    plt.close()

    print(f"  Saved to {pdf_output_path}")

print("Done.")
