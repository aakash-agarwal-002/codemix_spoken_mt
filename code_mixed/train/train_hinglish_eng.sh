CUDA_VISIBLE_DEVICES=0,1

data_dir=${1:-"../datasets/PHINC"}
model=${2:-"ai4bharat/indictrans2-indic-en-1B"}
output_dir=${3:-"../models/phinc_finetuned_hinglish_eng"}

python train_hinglish_eng.py \
    --data_dir $data_dir \
    --model $model \
    --output_dir $output_dir \
    --batch_size 32 \
    --num_train_epochs 10 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --logging_steps 200 \
    --save_steps 200 \
    --eval_steps 200 \
    --grad_accum_steps 2 \
    --warmup_ratio 0.1 \
    --max_grad_norm 1.0 \
    --label_smoothing 0.05 \
    --metric_for_best_model eval_BLEU \
    --greater_is_better \
    --patience 5 \
    --threshold 1e-3 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.2 \
    --lora_target_modules "q_proj,k_proj,v_proj,out_proj" \
    > ../logs/train_hinglish_eng.log 2>&1

