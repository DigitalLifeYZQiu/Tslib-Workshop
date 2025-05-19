export CUDA_VISIBLE_DEVICES=7

#ckpt_path="/data/zhanghaoran/projects/ltsm/experiments/bert/bert/lotsa_uea_bert_d768_l12_p36_n512_weight_p25_wo_revin_full/04012025_193158/ckpt/bert/bert/lotsa_uea_bert_d768_l12_p36_n512_weight_p25_wo_revin_full-epoch=14.ckpt"
#ckpt_path="/data/zhanghaoran/projects/ltsm/experiments/bert/bert/lotsa_uea_bert_d768_l12_p4_n512_weight_p50_wo_revin_full/07012025_015048/ckpt/bert/bert/lotsa_uea_bert_d768_l12_p4_n512_weight_p50_wo_revin_full-epoch=11.ckpt"
ckpt_path="/data/zhanghaoran/projects/ltsm/experiments/bert/bert/lotsa_uea_bert_d256_l4_p4_n512_weight_p25_wo_revin_wo_cls_long/26032025_014213/ckpt/bert/bert/lotsa_uea_bert_d256_l4_p4_n512_weight_p25_wo_revin_wo_cls_long-epoch=01.ckpt"
for seq_len in 40
do
for mask_rate in 0
do
for lr in 0.00001
do
for anomaly_ratio in 1
do
echo mask rate $mask_rate
python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --ckpt_path $ckpt_path \
  --root_path ./dataset/SWaT \
  --model_id SWAT \
  --model TimeBert \
  --data SWAT \
  --features M \
  --seq_len $seq_len \
  --pred_len 0 \
  --batch_size 32 \
  --patch_len 4 \
  --enc_in 51 \
  --train_epochs 20 \
  --stride 40 \
  --anomaly_ratio $anomaly_ratio \
  --learning_rate $lr \
  --lradj cosine \
  --ad_mask_type random \
  --mask_rate $mask_rate \
  --date_record \
  --not_use_variate_token
done
done
done
done
# > scriptsTimeBert/sota/TimeBert_SWAT.log && bash scriptsTimeBert/sota/TimeBert_SWAT.sh 2>&1 | tee -a scriptsTimeBert/sota/TimeBert_SWAT.log