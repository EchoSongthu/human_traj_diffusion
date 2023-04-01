CUDA_VISIBLE_DEVICES=0,5,6,7 python train_run.py --experiment  e2e-back \
--app "--init_emb /data/zmy/human_traj_diffusion/improved-diffusion/diffusion_models/exp58 --n_embd 16 --learned_emb yes " \
--notes "full_multi_sqrt_16" --epoch 200  --bsz 200 \
--dataset_name "mobile" --exp_n "home"

# augment
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_run.py --experiment  e2e-back \
--app "--init_emb /data/zmy/human_traj_diffusion/improved-diffusion/diffusion_models/exp58 --n_embd 16 --learned_emb yes " \
--notes "full_multi_sqrt_16" --epoch 150  --bsz 200 \
--dataset_name "mobile" --exp_n "home" --augment True 

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_run.py --experiment  e2e-back \
--app "--init_emb /data/zmy/human_traj_diffusion/improved-diffusion/diffusion_models/exp58 --n_embd 16 --learned_emb yes " \
--notes "full_multi_sqrt_16" --epoch 150  --bsz 200 \
--dataset_name "mobile" --exp_n "age" --augment True 


# tencent
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_run.py --experiment  e2e-back \
--app "--init_emb /data/zmy/human_traj_diffusion/improved-diffusion/diffusion_models/exp57 --n_embd 16 --learned_emb yes " \
--notes "full_multi_sqrt_16" --epoch 200  --bsz 250 \
--dataset_name "tencent" --exp_n "gender"


