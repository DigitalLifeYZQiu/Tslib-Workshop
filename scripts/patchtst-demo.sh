export CUDA_VISIBLE_DEVICES=0

model_name=PatchTST
for data in ETTh1; do
  for seq_len in 90 180;do
    for pred_len in 15; do
      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ../DATA/ETT-small/ \
        --data_path ${data}.csv \
        --model_id ${data}_sl${seq_len}_pl${pred_len} \
        --model $model_name \
        --data ${data} \
        --features MS \
        --seq_len ${seq_len} \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 3 \
        --d_layers 1 \
        --d_model 512 \
        --d_ff 1024 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --patch_len 15 \
        --stride 15 \
        --batch_size 512
    done
  done
done