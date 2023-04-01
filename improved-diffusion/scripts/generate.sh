exp=59

echo "..."
for((generate_name=600000;generate_name<1000000;generate_name=generate_name+50000))
do
python scripts/text_sample.py \
--model_path "diffusion_models/exp${exp}/ema_0.9999_${generate_name}.pt" \
--num_samples 1000 --top_p 1.0 --out_dir genout --eta 1. --gpu 1 \
--new_diffusion_steps 200 --batch_size 150
done
echo "done!!!"