export CUDA_VISIBLE_DEVICES=6

#ckpt_path="/data/zhanghaoran/projects/ltsm/experiments/bert/bert/lotsa_uea_bert_d768_l12_p36_n512_weight_p25_wo_revin_full/04012025_193158/ckpt/bert/bert/lotsa_uea_bert_d768_l12_p36_n512_weight_p25_wo_revin_full-epoch=14.ckpt"
ckpt_path="/data/zhanghaoran/projects/ltsm/experiments/bert/bert/lotsa_uea_bert_d768_l12_p4_n512_weight_p50_wo_revin_full/07012025_015048/ckpt/bert/bert/lotsa_uea_bert_d768_l12_p4_n512_weight_p50_wo_revin_full-epoch=11.ckpt"
for seq_len in 40
do
for mask_rate in 0.25 0.5 0.75
do
echo mask rate $mask_rate
python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --ckpt_path $ckpt_path \
  --root_path ./dataset/SMD \
  --model_id SMD_Original \
  --model TimeBert \
  --data SMD_Original \
  --features M \
  --seq_len $seq_len \
  --pred_len 0 \
  --batch_size 32 \
  --patch_len 4 \
  --enc_in 38 \
  --train_epochs 20 \
  --stride 1 \
  --anomaly_ratio 0.5 \
  --freeze_patch_encoder \
  --ad_mask_type random \
  --mask_rate $mask_rate \
  --date_record
done
done
#  > scriptsTimeBert/sota/TimeBert_SMD_freeze.log && bash scriptsTimeBert/sota/TimeBert_SMD.sh 2>&1 | tee -a scriptsTimeBert/fullft-1/TimeBert_SMD.log