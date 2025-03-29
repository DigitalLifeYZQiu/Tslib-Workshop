export CUDA_VISIBLE_DEVICES=0

model_name=torch_ARIMA_NLL
data_name=ETTh1

for pred_len in 96 192 336 720; do
python -u run.py \
  --task_name test_time_forecast \
  --is_training 0 \
  --root_path ../DATA/ETT-small/ \
  --data_path ${data_name}.csv \
  --model_id ${data_name}_96_${pred_len} \
  --model ${model_name} \
  --data ${data_name} \
  --features S \
  --batch_size 32 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len ${pred_len} \
  --des 'Exp' \
  --train_epochs 10 \
  --learning_rate 0.001 \
  --p 12 \
  --d 1 \
  --q 1 \
  --itr 1
done