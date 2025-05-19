export CUDA_VISIBLE_DEVICES=7

model_name=ARIMAlinMK3
data_name=m4

mkdir -p logs/${model_name}/
#logger=logs/${model_name}/${model_name}_${data_name}.log
#> "${logger}"

for lr in 1e-3 1e-4 1e-5;do
for sample_steps in 1 10 100 1000 10000;do
logger=logs/${model_name}/${model_name}_${data_name}_lr${lr}_sps${sample_steps}.log
> "${logger}"
for input_zoom in 2 3 4 5 6 7;do
for seasonal_patterns in Monthly Yearly Quarterly Daily Weekly Hourly;do
python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ../DATA/${data_name}/ \
  --seasonal_patterns ${seasonal_patterns} \
  --model_id ${data_name}_${seasonal_patterns} \
  --model ${model_name} \
  --data ${data_name} \
  --features M \
  --factor 3 \
  --batch_size 128 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --top_k 5 \
  --des 'Exp' \
  --train_epochs 100 \
  --patience 15 \
  --learning_rate ${lr} \
  --sample_steps ${sample_steps} \
  --loss 'SMAPE' \
  --input_zoom ${input_zoom} \
  --itr 1
done
done   2>&1 | tee -a "${logger}"
done
done
