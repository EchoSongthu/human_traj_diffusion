/data/zmy/human_traj_diffusion/improved-diffusion/scripts/run_clm.py             --output_dir=classifier_models/gender_mobile             --model_name_or_path=gpt2             --tokenizer_name=gpt2             --batch_size 200             --per_device_eval_batch_size 200             --save_steps 50000             --num_train_epochs 300             --do_train --eval_steps 10000 --evaluation_strategy steps             --do_eval --dataloader_num_workers 4             --save_total_limit 1             --overwrite_output_dir              --logging_dir classifier_models/runs/gender_mobile             --block_size 100              --disable_tqdm True --model_type gpt2             --gradient_accumulation_steps 1 --exp_n gender --experiment e2e-back --seed 101  --dataset_name=mobile --augment=False --dataset_config_name wikitext-103-raw-v1 --task wp --init_emb /data/zmy/human_traj_diffusion/improved-diffusion/diffusion_models/exp58 --n_embd 16 --learned_emb yes 
