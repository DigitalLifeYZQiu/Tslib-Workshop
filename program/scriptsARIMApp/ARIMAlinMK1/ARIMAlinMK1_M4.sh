#export CUDA_VISIBLE_DEVICES=0

model_name=ARIMAlinMK1
data_name=m4

logger=logs/${model_name}/${model_name}_${data_name}.log
> "${logger}"

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
  --batch_size 32 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --top_k 5 \
  --des 'Exp' \
  --train_epochs 20 \
  --learning_rate 0.001 \
  --loss 'SMAPE' \
  --p 12 \
  --d 1 \
  --q 1 \
  --itr 1
done  2>&1 | tee -a "${logger}"
