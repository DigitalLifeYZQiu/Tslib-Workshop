#export CUDA_VISIBLE_DEVICES=2

model_name=moment
ckpt_path=/home/zhanghaoran/projects/ltsm/lotsa_uea_bert_d768_l12_p24_n512_weight_p25_wo_revin_full-epoch=03.ckpt
d_model=256
d_ff=512
e_layers=4
patch_len=24
export HF_ENDPOINT=https://hf-mirror.com

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/MSL \
  --model_id MSL \
  --model Moment \
  --data MSL \
  --features M \
  --seq_len 192 \
  --pred_len 0 \
  --d_model 8 \
  --d_ff 16 \
  --e_layers 1 \
  --enc_in 55 \
  --c_out 55 \
  --top_k 3 \
  --anomaly_ratio 1 \
  --batch_size 64 \
  --train_epochs 1 \
  --patch_len 8 \
  --mask_rate 0.25

  #!/bin/sh

#export CUDA_VISIBLE_DEVICES=3
#
#model_name=moment
#ckpt_path=/home/zhanghaoran/projects/ltsm/lotsa_uea_bert_d768_l12_p24_n512_weight_p25_wo_revin_full-epoch=03.ckpt
#d_model=256
#d_ff=512
#e_layers=4
#patch_len=24
#
#export HF_ENDPOINT=https://hf-mirror.com
#
#for subset_rand_ratio in 1
#do
#for data in ETTh1
#do
#for mask_rate in 0.125 0.25 0.375 0.5
#do
#python -u run.py \
#  --task_name imputation \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --model_id $data\_mask_$mask_rate \
#  --mask_rate $mask_rate \
#  --model $model_name \
#  --ckpt_path $ckpt_path \
#  --data_path $data.csv \
#  --data $data \
#  --features M \
#  --seq_len 192 \
#  --label_len 0 \
#  --pred_len 192 \
#  --patch_len $patch_len \
#  --e_layers $e_layers \
#  --factor 3 \
#  --batch_size 16 \
#  --d_model $d_model \
#  --d_ff $d_ff \
#  --des 'Exp' \
#  --itr 1 \
#  --subset_rand_ratio $subset_rand_ratio \
#  --learning_rate 1e-5 \
#  --use_finetune
#done
#done
#done