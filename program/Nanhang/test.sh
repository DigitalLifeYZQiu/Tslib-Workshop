export CUDA_VISIBLE_DEVICES=0

model_name=PatchTST
#model_name=TTP_JointEmbedding
#model_name=TTP_noCausalAttn
#model_name=TTP

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /data/linjiafeng/Time-Series-Library/dataset/HPV_LEFT \
  --model_id aircraft_96 \
  --model $model_name \
  --data Aircraft \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 12 \
  --dec_in 12 \
  --c_out 12 \
  --factor 3 \
  --des 'Exp' \
  --patch_len 16 \
  --n_heads 8 \
  --itr 1