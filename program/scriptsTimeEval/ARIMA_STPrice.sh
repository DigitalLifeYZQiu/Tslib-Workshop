#export CUDA_VISIBLE_DEVICES=2

model_name=ARIMA

# ETTh1
for seq_len in 96
do
for pred_len in 16
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path /storage/dataset/DSET6f627eef49fb99e662eaa9f79a0d \
  --data_path ./Xiexindata/spot_trading/spot_trading_price.csv \
  --model_id STPrice_$seq_len_$pred_len \
  --model $model_name \
  --data STPrice \
  --features S \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --des 'Exp' \
  --target '电价(元/MWh)' \
  --inverse \
  --itr 1
done
done


for seq_len in 96
do
for pred_len in 16
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path /storage/dataset/DSET6f627eef49fb99e662eaa9f79a0d \
  --data_path ./Xiexindata/spot_trading/spot_trading_price.csv \
  --model_id STPrice_$seq_len_$pred_len \
  --model $model_name \
  --data STPrice \
  --features S \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --des 'Exp' \
  --target '电价(元/MWh)' \
  --itr 1
done
done