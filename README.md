# Code-Mixed and Spoken Language Machine Translation

A comprehensive project for fine-tuning transformer-based models to perform neural machine translation on **code-mixed text** (English-Hinglish) and **spoken language transcripts** (English-Hindi), with emphasis on handling Indian language scripts and transliteration.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Key Features](#key-features)
- [Dependencies](#dependencies)

---

## Overview

This project addresses machine translation challenges for Indian languages, specifically:

1. **Code-Mixed Translation**: Translating code-mixed text (Hinglish - English script mixed with Hindi words) to English and vice versa
2. **Spoken Language Translation**: Translating spoken English to Hindi and spoken Hindi to English based on TED talks transcripts

The project leverages fine-tuned versions of the [IndicTrans2](https://ai4bharat.iitm.ac.in/indictrans2/) model using **Parameter-Efficient Fine-Tuning (PEFT)** with Low-Rank Adaptation (LoRA) for efficient adaptation to these specific tasks.

### Key Metrics
- **Evaluation**: BLEU score and token-level accuracy
- **Models**: Base models from AI4Bharat's IndicTrans2 (1B parameters)
- **Fine-tuning**: LoRA-based PEFT approach
- **Languages**: English ↔ Hindi, English ↔ Hinglish

---

## Project Structure

```
codemix_spoken_mt/
├── code_mixed/                          # Code-mixed translation task
│   ├── train/
│   │   ├── train_eng_hinglish.py       # Fine-tune for EN→Hinglish
│   │   ├── train_eng_hinglish.sh       # Training script
│   │   ├── train_hinglish_eng.py       # Fine-tune for Hinglish→EN
│   │   └── train_hinglish_eng.sh       # Training script
│   ├── eval/
│   │   ├── eval_eng_hinglish.py        # Evaluate EN→Hinglish translations
│   │   ├── eval_hinglish_eng.py        # Evaluate Hinglish→EN translations
│   │   └── eval_*.sh                   # Evaluation shell scripts
│   ├── datasets_final/                 # Processed datasets
│   │   ├── PHINC/
│   │   ├── comi_lingua_mt/
│   │   ├── english_hinglish_top/
│   │   ├── hinge/
│   │   ├── Lince_benchmark_mt/
│   │   └── stats/                      # Dataset statistics
│   ├── models/                         # Fine-tuned model checkpoints
│   │   ├── phinc_finetuned_eng_hinglish/
│   │   └── phinc_finetuned_hinglish_eng/
│   ├── outputs/                        # Evaluation results (CSV)
│   ├── plots/                          # Training metrics visualizations
│   ├── dataset_utils.py                # Dataset preprocessing utilities
│   ├── dataset_stat.py                 # Dataset statistics computation
│   └── plots.py                        # Visualization of training metrics
│
├── spoken/                             # Spoken language translation task
│   ├── train/
│   │   ├── train_eng_hindi.py          # Fine-tune for EN→Hindi
│   │   ├── train_eng_hindi.sh          # Training script
│   │   ├── train_hindi_eng.py          # Fine-tune for Hindi→EN
│   │   └── train_hindi_eng.sh          # Training script
│   ├── eval/
│   │   ├── eval_eng_hindi.py           # Evaluate EN→Hindi translations
│   │   ├── eval_hindi_eng.py           # Evaluate Hindi→EN translations
│   │   └── eval_*.sh                   # Evaluation shell scripts
│   ├── datasets/
│   │   ├── TED2020/                    # TED talks corpus
│   │   └── stats/                      # Dataset statistics
│   ├── models/                         # Fine-tuned model checkpoints
│   │   ├── TED2020_finetuned_eng_hindi/
│   │   └── TED2020_finetuned_hindi_eng/
│   ├── outputs/                        # Evaluation results (CSV)
│   ├── outputs_clean/                  # Cleaned evaluation results
│   ├── plots/                          # Training metrics visualizations
│   ├── dataset_utils.py                # Dataset preprocessing utilities
│   ├── dataset_stat.py                 # Dataset statistics computation
│   ├── plots.py                        # Visualization of training metrics
│   └── remove_blanks.py                # Data cleaning utility
│
├── itv2.yml                            # Conda environment configuration
├── requirments.txt                     # Python package dependencies
└── README.md                           # Project documentation
```

---

## Datasets

### Code-Mixed Datasets

| Dataset | Source | Task | Split |
|---------|--------|------|-------|
| **PHINC** | LingoIITGN | EN↔Hinglish | train/val/test |


**Key Features:**
- Transliteration support (Hindi script ↔ Latin script for Hinglish)
- Automatic train/validation/test splits
- Statistics includes word count distributions, character lengths

### Spoken Language Dataset

| Dataset | Source | Task | Content |
|---------|--------|------|---------|
| **TED2020** | OPUS Corpus | EN↔Hindi | TED talk transcripts and translations |

**Download Details:**
- Automatically downloaded from: `https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/en-hi.txt.zip`
- English and Hindi parallel sentences from TED talks

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd codemix_spoken_mt
```

### 2. Set Up Conda Environment

```bash
conda env create -f itv2.yml
conda activate itv2
```

### 3. Install Via pip (if needed)

```bash
pip install -r requirments.txt
```

### Key Dependencies

```
Python 3.10
torch (GPU-enabled recommended)
transformers
datasets
peft (for LoRA fine-tuning)
IndicTransToolkit
ai4bharat-transliteration
fairseq
evaluate
```

---

## Usage

### Training

#### Code-Mixed Translation (English → Hinglish)

```bash
cd code_mixed/train
python train_eng_hinglish.py \
    --model ai4bharat/indictrans2-en-indic-1B \
    --data_dir ../datasets/PHINC \
    --output_dir ../models/phinc_finetuned_eng_hinglish \
    --batch_size 8 \
    --num_train_epochs 3 \
    --learning_rate 3e-4
```

Or use the provided shell script:

```bash
bash train_eng_hinglish.sh
```

#### Code-Mixed Translation (Hinglish → English)

```bash
cd code_mixed/train
python train_hinglish_eng.py \
    --model ai4bharat/indictrans2-en-indic-1B \
    --data_dir ../datasets/PHINC \
    --output_dir ../models/phinc_finetuned_hinglish_eng \
    --batch_size 8 \
    --num_train_epochs 3 \
    --learning_rate 3e-4
```

#### Spoken Language Translation (English → Hindi)

```bash
cd spoken/train
python train_eng_hindi.py \
    --model ai4bharat/indictrans2-en-indic-1B \
    --data_dir ../datasets/TED2020 \
    --output_dir ../models/TED2020_finetuned_eng_hindi \
    --batch_size 8 \
    --num_train_epochs 3 \
    --learning_rate 3e-4
```

#### Spoken Language Translation (Hindi → English)

```bash
cd spoken/train
python train_hindi_eng.py \
    --model ai4bharat/indictrans2-en-indic-1B \
    --data_dir ../datasets/TED2020 \
    --output_dir ../models/TED2020_finetuned_hindi_eng \
    --batch_size 8 \
    --num_train_epochs 3 \
    --learning_rate 3e-4
```

### Evaluation & Translation

#### Evaluate Code-Mixed Models

```bash
cd code_mixed/eval

# Evaluate base model (EN→Hinglish)
python eval_eng_hinglish.py \
    --data_dir ../datasets/PHINC \
    --base_ckpt ai4bharat/indictrans2-en-indic-1B \
    --use_peft  # Use if PEFT-finetuned

# Evaluate fine-tuned model
python eval_eng_hinglish.py \
    --data_dir ../datasets/PHINC \
    --adapter_path ../models/phinc_finetuned_eng_hinglish/checkpoint-500 \
    --use_peft
```

#### Evaluate Spoken Language Models

```bash
cd spoken/eval

# Evaluate EN→Hindi translation
python eval_eng_hindi.py \
    --data_dir ../datasets/TED2020 \
    --adapter_path ../models/TED2020_finetuned_eng_hindi/checkpoint-500 \
    --use_peft
```

### Dataset Preparation & Statistics

#### Compute Dataset Statistics

```bash
cd code_mixed
python dataset_stat.py    # Computes word count, character length stats
```

#### Prepare Datasets

The `prepare_dataset()` function in `dataset_utils.py`:
- Loads datasets from HuggingFace or local disk
- Automatically handles train/val/test splits
- Adds transliteration columns for code-mixed text
- Caches processed datasets

---

## Model Architecture

### Base Model

- **Model**: `ai4bharat/indictrans2-en-indic-1B`
- **Architecture**: Transformer-based sequence-to-sequence model
- **Parameters**: ~1 Billion
- **Task**: Multilingual machine translation (supports 22+ Indian languages)
- **Source**: [AI4Bharat - IndicTrans2](https://ai4bharat.iitm.ac.in/indictrans2/)

### Fine-tuning Approach

**PEFT with LoRA (Low-Rank Adaptation)**

```
Rank (r): 8
Lora Alpha: 16
Lora Dropout: 0.05
Target Modules: 
  - q_proj (query projections)
  - v_proj (value projections)
  - k_proj (key projections)
  - out_proj (output projections)
```

**Benefits:**
- 10-100x reduction in trainable parameters
- Faster fine-tuning
- Memory-efficient
- Maintains base model performance while adapting to specific tasks

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 8 |
| Learning Rate | 3e-4 |
| Weight Decay | 0.01 |
| Epochs | 3 |
| Warmup Ratio | 0.0 |
| Gradient Accumulation Steps | 1 |
| Save Frequency | Every 500 steps |
| Evaluation Frequency | Every 500 steps |
| Early Stopping | Monitored |

### Pre/Post-processing

- **IndicProcessor**: Handles language-specific text preprocessing
- **XlitEngine**: Transliterates between Indian scripts (Devanagari ↔ Latin)
- **Tokenizer**: BPE-based tokenizer from HuggingFace transformers

---

## Results

### Output Files

#### Translation Results (CSV)

`outputs/` directories contain CSV files with:
- **english**: Original English text
- **hinglish/hindi**: Original target language text
- **hinglish_dev/hindi_dev**: Transliterated/preprocessed text
- **pred_hinglish_dev/pred_hindi**: Predictions in target script
- **pred_hinglish/pred_hindi**: Predictions in original script

#### Evaluation Metrics

Results tracked in:
- `en_to_hinglish_results_base.csv` / `_finetuned.csv`
- `hinglish_to_en_results_base.csv` / `_finetuned.csv`
- Similar files for English-Hindi translation

**Metrics Computed:**
- **BLEU Score**: Bilingual Evaluation Understudy metric
- **Token-level Accuracy**: Exact character/token matching
- **METEOR** (optional): metric for machine translation evaluation

#### Visualization

`plots/` directories contain:
- Training and evaluation loss curves
- BLEU score progression over epochs
- Model performance comparison (base vs. fine-tuned)

---

## Key Features

### 1. **Multi-directional Translation**
- English → Code-mixed (Hinglish)
- Code-mixed (Hinglish) → English
- English → Hindi
- Hindi → English

### 2. **Robust Data Processing**
- Automatic dataset discovery and loading from HuggingFace
- Parallel processing for statistics computation
- Automatic train/val/test splitting
- Data cleaning and validation

### 3. **Advanced Fine-tuning**
- Parameter-Efficient Fine-Tuning (PEFT) with LoRA
- Early stopping callback
- Mixed precision training support
- Gradient accumulation for larger effective batch sizes

### 4. **Transliteration Support**
- Hinglish ↔ Hindi script conversion using XlitEngine
- Automatic alignment of target scripts
- Beam search for transliteration (beam_width=5)

### 5. **Comprehensive Evaluation**
- BLEU score computation
- Batch-based inference with progress tracking
- Multiprocessing support for transliteration
- Results export to CSV format

### 6. **Utilities & Helpers**
- `dataset_utils.py`: Dataset loading and preprocessing
- `dataset_stat.py`: Statistical analysis of datasets
- `plots.py`: Visualization of training metrics
- `remove_blanks.py`: Data cleaning for results

---

## Important Notes

### GPU Requirements
- Fine-tuning: Recommended 24GB+ GPU memory (e.g., A100, RTX 3090)
- Inference: 8GB+ GPU memory sufficient for most cases
- CPU-only inference possible but significantly slower

### Transliteration
- XlitEngine uses CPU-based beam search (forced via `CUDA_VISIBLE_DEVICES=""`)
- Multiprocessing-friendly for parallel transliteration
- Beam width set to 5 for accuracy-speed trade-off

### HuggingFace Cache
- Set via `HF_HOME` environment variable
- Offline mode enabled for reproducibility
- Pre-download models before offline usage

### File Paths
- Training scripts expect relative paths from their directories
- Absolute paths recommended for production use
- Update `dataset_path` mapping in shell scripts as needed

---

## Environment Variables

```bash
# Set HuggingFace cache directory
export HF_HOME="/path/to/huggingface/cache"

# For offline mode (recommended)
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# GPU selection (if multiple GPUs available)
export CUDA_VISIBLE_DEVICES=0
```

---

## Troubleshooting

### Model Loading Issues
```bash
# Clear HuggingFace cache
rm -rf $HF_HOME

# Re-download models (requires internet)
unset TRANSFORMERS_OFFLINE
python -c "from transformers import AutoModel; AutoModel.from_pretrained('ai4bharat/indictrans2-en-indic-1B')"
```

### Out of Memory (OOM)
- Reduce `batch_size` (default: 8 → try 4 or 2)
- Enable gradient accumulation: `--grad_accum_steps 2`
- Use mixed precision: Add `--fp16` flag

### Transliteration Errors
- Ensure `indic-nlp-library` is properly installed
- Check XlitEngine language codes (e.g., "hi" for Hindi)
- Verify input text encoding (UTF-8 expected)

---

## Citation & References

If you use this project, please cite:

```bibtex
@misc{codemix_spoken_mt,
  title={Code-Mixed and Spoken Language Machine Translation},
  author={Your Name},
  year={2024}
}
```

### Related Work

- [IndicTrans2](https://ai4bharat.iitm.ac.in/indictrans2/) - Base models
- [PEFT Library](https://huggingface.co/docs/peft/) - LoRA implementation
- [Transformers](https://huggingface.co/transformers/) - Model architecture
- [Datasets](https://huggingface.co/datasets/) - Data handling

---

## Contact & Support

For issues or questions:
- Check the `logs/` directories for runtime logs
- Review training arguments in shell scripts
- Verify dataset paths exist before running
- Ensure conda environment is properly activated

---

## License

[Add your license here]

Last Updated: February 2026
