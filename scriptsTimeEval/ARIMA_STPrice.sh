export CUDA_VISIBLE_DEVICES=2

model_name=ARIMA

# ETTh1
for seq_len in 96
do
for pred_len in 16
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ../DATA/Xiexin/ \
  --data_path spot_trading_price.csv \
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
#
## ETTh2
#for seq_len in 96
#do
#for pred_len in 96 192 336 720
#do
#python -u run.py \
#  --task_name long_term_forecast \
#  --is_training 0 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTh2.csv \
#  --model_id ETTh2_$seq_len_$pred_len \
#  --model $model_name \
#  --data ETTh2 \
#  --features S \
#  --seq_len $seq_len \
#  --pred_len $pred_len \
#  --des 'Exp' \
#  --itr 1
#done
#done
#
## ETTm1
#for seq_len in 96
#do
#for pred_len in 96 192 336 720
#do
#python -u run.py \
#  --task_name long_term_forecast \
#  --is_training 0 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTm1.csv \
#  --model_id ETTm1_$seq_len_$pred_len \
#  --model $model_name \
#  --data ETTm1 \
#  --features S \
#  --seq_len $seq_len \
#  --pred_len $pred_len \
#  --des 'Exp' \
#  --itr 1
#done
#done
#
## ETTm2
#for seq_len in 96
#do
#for pred_len in 96 192 336 720
#do
#python -u run.py \
#  --task_name long_term_forecast \
#  --is_training 0 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTm2.csv \
#  --model_id ETTm2_$seq_len_$pred_len \
#  --model $model_name \
#  --data ETTm2 \
#  --features S \
#  --seq_len $seq_len \
#  --pred_len $pred_len \
#  --des 'Exp' \
#  --itr 1
#done
#done
#
##ECL
#for seq_len in 96
#do
#for pred_len in 96 192 336 720
#do
#python -u run.py \
#  --task_name long_term_forecast \
#  --is_training 0 \
#  --root_path ./dataset/electricity/ \
#  --data_path electricity.csv \
#  --model_id ECL_$seq_len_$pred_len \
#  --model $model_name \
#  --data custom \
#  --features S \
#  --seq_len $seq_len \
#  --pred_len $pred_len \
#  --des 'Exp' \
#  --itr 1
#done
#done
#
##Traffic
#for seq_len in 96
#do
#for pred_len in 96 192 336 720
#do
#python -u run.py \
#  --task_name long_term_forecast \
#  --is_training 0 \
#  --root_path ./dataset/traffic/ \
#  --data_path traffic.csv \
#  --model_id traffic_$seq_len_$pred_len \
#  --model $model_name \
#  --data custom \
#  --features S \
#  --seq_len $seq_len \
#  --pred_len $pred_len \
#  --des 'Exp' \
#  --itr 1
#done
#done
#
##Weather
#for seq_len in 96
#do
#for pred_len in 96 192 336 720
#do
#python -u run.py \
#  --task_name long_term_forecast \
#  --is_training 0 \
#  --root_path ./dataset/weather/ \
#  --data_path weather.csv \
#  --model_id weather_$seq_len_$pred_len \
#  --model $model_name \
#  --data custom \
#  --features S \
#  --seq_len $seq_len \
#  --pred_len $pred_len \
#  --des 'Exp' \
#  --itr 1
#done
#done
#
## Exchange
#for seq_len in 96
#do
#for pred_len in 96 192 336 720
#do
#python -u run.py \
#  --task_name long_term_forecast \
#  --is_training 0 \
#  --root_path ./dataset/exchange_rate/ \
#  --data_path exchange_rate.csv \
#  --model_id Exchange_$seq_len_$pred_len \
#  --model $model_name \
#  --data custom \
#  --features S \
#  --seq_len $seq_len \
#  --pred_len $pred_len \
#  --des 'Exp' \
#  --itr 1
#done
#done