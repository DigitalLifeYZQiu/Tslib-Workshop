export CUDA_VISIBLE_DEVICES=0


dataset_dir="./dataset/UCR_Anomaly_FullData"
counter=0

for file_path in "$dataset_dir"/*
do
data=$(basename "$file_path")
# data="004_UCR_Anomaly_DISTORTEDBIDMC1_2500_5400_5600.txt"
((counter++))
echo $counter
python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/UCR_Anomaly_FullData \
  --data_path $data \
  --model_id UCRA_$data \
  --model DLinear \
  --data UCRA \
  --features M \
  --seq_len 100 \
  --pred_len 100 \
  --d_model 128 \
  --d_ff 128 \
  --e_layers 3 \
  --enc_in 1 \
  --c_out 1 \
  --top_k 3 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 10
  if ((counter>4)); then
    break
  fi
done