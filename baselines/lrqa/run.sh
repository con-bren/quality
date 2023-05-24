
current_dir=$(pwd)
export PYTHONPATH=${current_dir}/:${PYTHONPATH} 

python lrqa/run_lrqa.py \
    --model_name_or_path gpt2 \
    --model_mode generation \
    --task_name custom \
    --task_base_path ../../data/test \
    --output_dir exp \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_strategy steps \
    --load_best_model_at_end \
    --num_train_epochs 1 \
    --dataloader_num_workers 0
