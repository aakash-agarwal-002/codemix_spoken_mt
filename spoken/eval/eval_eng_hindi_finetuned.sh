 CUDA_VISIBLE_DEVICES=0,1

data_dir=${1:-"../datasets/TED2020"}
base_ckpt=${2:-"ai4bharat/indictrans2-en-indic-1B"}
adapter_path=${3:-"../models/TED2020_finetuned_eng_hindi"}
limit_n=${4:-None}
batch_size=${5:-64}
use_peft=${6:-true}
output_dir=${7:-"../outputs"}

log_file="../logs/eval_eng_hindi_finetuned.log"

if [ "$use_peft" = true ]; then
    python eval_eng_hindi.py \
        --data_dir "$data_dir" \
        --base_ckpt "$base_ckpt" \
        --adapter_path "$adapter_path" \
        --use_peft \
        --batch_size "$batch_size" \
        --limit_n "$limit_n" \
        --output_dir "$output_dir" \
        > "$log_file" 2>&1
else
    python eval_eng_hindi.py \
        --data_dir "$data_dir" \
        --base_ckpt "$base_ckpt" \
        --batch_size "$batch_size" \
        --limit_n "$limit_n" \
        --output_dir "$output_dir" \
        > "$log_file" 2>&1
fi

echo "English → Hinglish evaluation complete."
echo "Log saved to $log_file"
