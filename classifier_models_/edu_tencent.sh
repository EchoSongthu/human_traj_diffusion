/data/zmy/human_traj_diffusion/improved-diffusion/scripts/run_clm.py             --output_dir=classifier_models/edu_tencent             --model_name_or_path=gpt2             --tokenizer_name=gpt2             --batch_size 250             --per_device_eval_batch_size 250             --save_steps 50000             --num_train_epochs 200             --do_train --eval_steps 10000 --evaluation_strategy steps             --do_eval --dataloader_num_workers 4             --save_total_limit 1             --overwrite_output_dir              --logging_dir classifier_models/runs/edu_tencent             --block_size 100              --disable_tqdm True --model_type gpt2             --gradient_accumulation_steps 1 --exp_n edu --experiment e2e-back --seed 101  --dataset_name=tencent --augment=False --dataset_config_name wikitext-103-raw-v1 --task wp --init_emb /data/zmy/human_traj_diffusion/improved-diffusion/diffusion_models/exp57 --n_embd 16 --learned_emb yes 
