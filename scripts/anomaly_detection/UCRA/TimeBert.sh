export CUDA_VISIBLE_DEVICES=2

dataset_dir="./dataset/UCR_Anomaly_FullData"
counter=0
ckpt_path="/data/zhanghaoran/projects/ltsm/experiments/bert/bert/lotsa_uea_bert_d768_l12_p36_n512_weight_p25_wo_revin_full/04012025_193158/ckpt/bert/bert/lotsa_uea_bert_d768_l12_p36_n512_weight_p25_wo_revin_full-epoch=14.ckpt"

for file_path in "$dataset_dir"/*
do
data=$(basename "$file_path")
# data="004_UCR_Anomaly_DISTORTEDBIDMC1_2500_5400_5600.txt"
((counter++))
echo $counter
python -u run.py \
  --task_name anomaly_detection_ltm \
  --is_training 1 \
  --ckpt_path $ckpt_path \
  --root_path ./dataset/UCR_Anomaly_FullData \
  --data_path $data \
  --model_id UCRA_$data \
  --model TimeBert \
  --data UCRA \
  --features M \
  --seq_len 360 \
  --pred_len 0 \
  --batch_size 1024 \
  --train_epochs 10 \
  --date_record
  if ((counter>4)); then
    break
  fi
done