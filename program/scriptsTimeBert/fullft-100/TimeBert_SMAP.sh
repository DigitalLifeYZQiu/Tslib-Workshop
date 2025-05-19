export CUDA_VISIBLE_DEVICES=0

ckpt_path="/data/zhanghaoran/projects/ltsm/experiments/bert/bert/lotsa_uea_bert_d768_l12_p36_n512_weight_p25_wo_revin_full/04012025_193158/ckpt/bert/bert/lotsa_uea_bert_d768_l12_p36_n512_weight_p25_wo_revin_full-epoch=14.ckpt"

python -u run.py \
  --task_name anomaly_detection_ltm \
  --is_training 0 \
  --ckpt_path $ckpt_path \
  --root_path ./dataset/SMAP \
  --model_id SMAP \
  --model TimeBert \
  --data SMAP \
  --features M \
  --seq_len 360 \
  --pred_len 0 \
  --batch_size 16 \
  --train_epochs 10 \
  --stride 100 \
  --date_record