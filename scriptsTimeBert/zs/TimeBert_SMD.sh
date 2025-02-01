export CUDA_VISIBLE_DEVICES=7

#ckpt_path="/data/zhanghaoran/projects/ltsm/experiments/bert/bert/lotsa_uea_bert_d768_l12_p36_n512_weight_p25_wo_revin_full/04012025_193158/ckpt/bert/bert/lotsa_uea_bert_d768_l12_p36_n512_weight_p25_wo_revin_full-epoch=14.ckpt"
ckpt_path="/data/zhanghaoran/projects/ltsm/experiments/bert/bert/lotsa_uea_bert_d768_l12_p4_n512_weight_p50_wo_revin_full/07012025_015048/ckpt/bert/bert/lotsa_uea_bert_d768_l12_p4_n512_weight_p50_wo_revin_full-epoch=11.ckpt"

python -u run.py \
  --task_name anomaly_detection_ltm \
  --is_training 0 \
  --ckpt_path $ckpt_path \
  --root_path ./dataset/SMD \
  --model_id SMD \
  --model TimeBert \
  --data SMD \
  --features M \
  --seq_len 40 \
  --patch_len 4 \
  --enc_in 38 \
  --pred_len 0 \
  --batch_size 16 \
  --train_epochs 10 \
  --stride 1 \
  --date_record