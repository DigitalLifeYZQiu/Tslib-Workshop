export CUDA_VISIBLE_DEVICES=5

ckpt_path="/data/zhanghaoran/projects/ltsm/experiments/bert/bert/lotsa_uea_bert_d768_l12_p36_n512_weight_p25_wo_revin_full/04012025_193158/ckpt/bert/bert/lotsa_uea_bert_d768_l12_p36_n512_weight_p25_wo_revin_full-epoch=14.ckpt"

for seq_len in 360 720
do
for mask_rate in 0.25 0.5 0.75
do
echo mask rate $mask_rate
python -u run.py \
  --task_name anomaly_detection_ltm \
  --is_training 1 \
  --ckpt_path $ckpt_path \
  --root_path ./dataset/PSM \
  --model_id PSM \
  --model TimeBert \
  --data PSM \
  --features M \
  --seq_len $seq_len \
  --pred_len 0 \
  --batch_size 16 \
  --patch_len 36 \
  --train_epochs 20 \
  --stride 1 \
  --ad_mask_type random \
  --mask_rate $mask_rate \
  --freeze_patch_encoder \
  --date_record
done
done
#  > scriptsTimeBert/fullft-1/TimeBert_PSM.log && bash scriptsTimeBert/fullft-1/TimeBert_PSM.sh 2>&1 | tee -a scriptsTimeBert/fullft-1/TimeBert_PSM.log