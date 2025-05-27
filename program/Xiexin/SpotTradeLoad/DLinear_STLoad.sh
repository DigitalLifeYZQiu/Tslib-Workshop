#export CUDA_VISIBLE_DEVICES=2

model_name=DLinear
data=Xiexin_SpotTradeLoad
root_path=/data/qiuyunzhong/DATA/Xiexin
data_path=guangdong_spot_trading_electricity_2025.csv

# ETTh1
for seq_len in 1440
do
for pred_len in 40
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id $data_$seq_len_$pred_len \
  --model $model_name \
  --data $data \
  --features S \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --des 'Exp' \
  --target electricity \
  --inverse \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id $data_$seq_len_$pred_len \
  --model $model_name \
  --data $data \
  --features S \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --des 'Exp' \
  --target electricity \
  --itr 1
done
done