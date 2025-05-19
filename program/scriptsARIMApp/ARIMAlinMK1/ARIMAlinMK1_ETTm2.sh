#export CUDA_VISIBLE_DEVICES=0

model_name=ARIMAlinMK1
data_name=ETTm2

for pred_len in 96 192 336 720; do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../DATA/ETT-small/ \
  --data_path ${data_name}.csv \
  --model_id ${data_name}_96_${pred_len} \
  --model ${model_name} \
  --data ${data_name} \
  --features S \
  --batch_size 32 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len ${pred_len} \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --train_epochs 20 \
  --learning_rate 0.001 \
  --p 12 \
  --d 1 \
  --q 1 \
  --itr 1
done