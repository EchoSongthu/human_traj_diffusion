# condition
# mobile
CUDA_VISIBLE_DEVICES=1,0 python infill2.py \
--model_path /data/zmy/human_traj_diffusion/improved-diffusion/diffusion_models/exp58/ema_0.9999_170000.pt \
--eval_task_ 'control_attribute' --use_ddim True  \
--notes "tree_full_adagrad" --eta 1. --verbose pipe --dataset 'mobile' \
--exp_n 4 --k 5 --coef 0.001 --num_samples 50 --batch_size 50 --control "home"

CUDA_VISIBLE_DEVICES=4,5 python infill2.py \
--model_path /data/zmy/human_traj_diffusion/improved-diffusion/diffusion_models/exp58/ema_0.9999_170000.pt \
--eval_task_ 'control_attribute' --use_ddim True  \
--notes "tree_full_adagrad" --eta 1. --verbose pipe --dataset 'mobile' \
--exp_n 0 --k 0 --coef 0.01 --num_samples 50 --batch_size 50 --control "home"


# 新模型
python infill2.py \
--model_path /data/zmy/human_traj_diffusion/improved-diffusion/diffusion_models/exp66/ema_0.9999_150000.pt \
--eval_task_ 'control_attribute' --use_ddim True  \
--notes "tree_full_adagrad" --eta 1. --verbose pipe --dataset 'mobile' \
--exp_n 15 --k 0 --coef 0.01 --num_samples 100 --batch_size 100 --control "gender"


# augment
CUDA_VISIBLE_DEVICES=0,6 python infill2.py \
--model_path /data/zmy/human_traj_diffusion/improved-diffusion/diffusion_models/exp58/ema_0.9999_170000.pt \
--eval_task_ 'control_attribute' --use_ddim True  \
--notes "tree_full_adagrad" --eta 1. --verbose pipe --dataset 'mobile' \
--exp_n 12 --k 3 --coef 0.01 --num_samples 50 --batch_size 50 --control "edu" --if_augment True


CUDA_VISIBLE_DEVICES=0,1,2 python infill2.py \
--model_path /data/zmy/human_traj_diffusion/improved-diffusion/diffusion_models/exp58/ema_0.9999_170000.pt \
--eval_task_ 'control_attribute' --use_ddim True  \
--notes "tree_full_adagrad" --eta 1. --verbose pipe --dataset 'mobile' \
--exp_n 12 --k 3 --coef 0.01 --num_samples 100 --batch_size 100 --control "edu" --if_filter True


CUDA_VISIBLE_DEVICES=7,0 python infill2.py \
--model_path /data/zmy/human_traj_diffusion/improved-diffusion/diffusion_models/exp58/ema_0.9999_170000.pt \
--eval_task_ 'control_attribute' --use_ddim True  \
--notes "tree_full_adagrad" --eta 1. --verbose pipe --dataset 'mobile' \
--exp_n 0 --k 3 --coef 0.01 --num_samples 100 --batch_size 100 --control "gender"  --checkpoint 7000




CUDA_VISIBLE_DEVICES=7,6 python infill2.py \
--model_path /data/zmy/human_traj_diffusion/improved-diffusion/diffusion_models/exp70/ema_0.9999_225000.pt \
--eval_task_ 'control_attribute' --use_ddim True  \
--notes "tree_full_adagrad" --eta 1. --verbose pipe --dataset 'mobile' \
--exp_n 3 --k 3 --coef 0.01 --num_samples 100 --batch_size 100 --control "gender" --checkpoint 5000










--if_filter True
--if_augment True






# tencent
CUDA_VISIBLE_DEVICES=1,0,2,3 python infill2.py \
--model_path /data/zmy/human_traj_diffusion/improved-diffusion/diffusion_models/exp57/ema_0.9999_550000.pt \
--eval_task_ 'control_attribute' --use_ddim True  \
--notes "tree_full_adagrad" --eta 1. --verbose pipe --dataset 'tencent' \
--exp_n 0 --k 0 --coef 0.01 --num_samples 50 --batch_size 50 --control "age"


# 数组增强
CUDA_VISIBLE_DEVICES=2,3 python infill2.py \
--model_path /data/zmy/human_traj_diffusion/improved-diffusion/diffusion_models/exp58/ema_0.9999_170000.pt \
--eval_task_ 'control_attribute' --use_ddim True  \
--notes "tree_full_adagrad" --eta 1. --verbose pipe --dataset 'mobile' \
--num_samples 30 --k 3 --coef 0.01 --exp_n 8 --if_augment True \
--control "home" --batch_size 30  --file_name 2