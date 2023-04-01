CUDA_VISIBLE_DEVICES=4,5,6,7 python train_run.py --experiment  e2e-back \
--app "--init_emb /data/zmy/human_traj_diffusion/improved-diffusion/diffusion_models/exp58 --n_embd 16 --learned_emb yes " \
--notes "full_multi_sqrt_16" --epoch 250  --bsz 200 \
--dataset_name "mobile" --exp_n "edu" --augment True 

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_run.py --experiment  e2e-back \
--app "--init_emb /data/zmy/human_traj_diffusion/improved-diffusion/diffusion_models/exp58 --n_embd 16 --learned_emb yes " \
--notes "full_multi_sqrt_16" --epoch 300  --bsz 200 \
--dataset_name "mobile" --exp_n "age"

CUDA_VISIBLE_DEVICES=4,5,6 python train_run.py --experiment  e2e-back \
--app "--init_emb /data/zmy/human_traj_diffusion/improved-diffusion/diffusion_models/exp58 --n_embd 16 --learned_emb yes " \
--notes "full_multi_sqrt_16" --epoch 300  --bsz 200 \
--dataset_name "mobile" --exp_n "gender"


CUDA_VISIBLE_DEVICES=0,1,2 python train_run.py --experiment  e2e-back \
--app "--init_emb /data/zmy/human_traj_diffusion/improved-diffusion/diffusion_models/exp58 --n_embd 16 --learned_emb yes " \
--notes "full_multi_sqrt_16" --epoch 300  --bsz 200 \
--dataset_name "mobile" --exp_n "gender"

CUDA_VISIBLE_DEVICES=3,4,5,6 python train_run.py --experiment  e2e-back \
--app "--init_emb /data/zmy/human_traj_diffusion/improved-diffusion/diffusion_models/exp70 --n_embd 16 --learned_emb yes " \
--notes "full_multi_sqrt_16" --epoch 400  --bsz 200 \
--dataset_name "mobile" --exp_n "gender"





# zombie!!!
CUDA_VISIBLE_DEVICES=3,4,5,6 python train_run.py --experiment  e2e-back \
--app "--init_emb /data/zmy/human_traj_diffusion/improved-diffusion/diffusion_models/exp58 --n_embd 16 --learned_emb yes " \
--notes "full_multi_sqrt_16" --epoch 3000000  --bsz 100 \
--dataset_name "mobile" --exp_n "none"