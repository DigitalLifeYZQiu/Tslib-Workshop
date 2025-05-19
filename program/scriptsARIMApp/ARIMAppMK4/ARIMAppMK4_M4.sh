export CUDA_VISIBLE_DEVICES=2

model_name=ARIMAppMK4
data_name=m4

mkdir -p logs/${model_name}/
logger=logs/${model_name}/${model_name}_${data_name}.log
> "${logger}"

for input_zoom in 2 3 4 5 6 7;do
echo "Executing input zoom ${input_zoom}"
for seasonal_patterns in Monthly Yearly Quarterly Daily Weekly Hourly;do
#for seasonal_patterns in Yearly;do
python -u run.py \
  --task_name short_term_forecast \
  --is_training 0 \
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
  --train_epochs 3 \
  --learning_rate 5e-4 \
  --loss "SMAPE" \
  --input_zoom ${input_zoom} \
  --p 12 \
  --d 1 \
  --q 12 \
  --itr 1
done
done  2>&1 | tee -a "${logger}"
# >>  ${logger} 2>&1 &
