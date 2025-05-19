export CUDA_VISIBLE_DEVICES=5

dataset_dir="../DATA/UCR_Anomaly_FullData"
counter=0

for file_path in "$dataset_dir"/*
do
data=$(basename "$file_path")
((counter++))
echo $counter
python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ${dataset_dir} \
  --data_path $data \
  --model_id UCRA_$data \
  --model TimesNet \
  --data UCRA \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 128 \
  --d_ff 128 \
  --e_layers 2 \
  --enc_in 1 \
  --c_out 1 \
  --top_k 3 \
  --anomaly_ratio 0.1 \
  --batch_size 128 \
  --train_epochs 1
  if ((counter>4)); then
    break
  fi
done