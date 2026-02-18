CUDA_VISIBLE_DEVICES=0,1

data_dir=${1:-"../datasets/PHINC"}
base_ckpt=${2:-"ai4bharat/indictrans2-indic-en-1B"}
adapter_path=${3:-"../models/phinc_finetuned_hinglish_eng"}
limit_n=${4:-None}
batch_size=${5:-64}
use_peft=${6:-false}
output_dir=${7:-"../outputs"}

log_file="../logs/eval_hinglish_eng_base.log"

if [ "$use_peft" = true ]; then
    python eval_hinglish_eng.py \
        --data_dir "$data_dir" \
        --base_ckpt "$base_ckpt" \
        --adapter_path "$adapter_path" \
        --use_peft \
        --batch_size "$batch_size" \
        --limit_n "$limit_n" \
        --output_dir "$output_dir" \
        > "$log_file" 2>&1
else
    python eval_hinglish_eng.py \
        --data_dir "$data_dir" \
        --base_ckpt "$base_ckpt" \
        --batch_size "$batch_size" \
        --limit_n "$limit_n" \
        --output_dir "$output_dir" \
        > "$log_file" 2>&1
fi

echo "Evaluation complete. Log saved to $log_file"
