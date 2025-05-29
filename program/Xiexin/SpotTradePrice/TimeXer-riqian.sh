export CUDA_VISIBLE_DEVICES=1

model_name=TimeXer
des='Timexer-MS-Xiexin'
seq_len=1440
pred_len=40
data=Xiexin_SpotTradePrice
d_model=512
d_ff=1024

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /data/qiuyunzhong/DATA/Xiexin/ \
  --data_path Actual-Price-riqian.csv \
  --model_id $data_$seq_len_$pred_len \
  --model $model_name \
  --data $data \
  --features MS \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 1 \
  --factor 3 \
  --enc_in 11 \
  --dec_in 11 \
  --c_out 11 \
  --des $des \
  --batch_size 32 \
  --learning_rate 5e-6 \
  --lradj cosine \
  --patience 5 \
  --train_epochs 30 \
  --d_model ${d_model} \
  --d_ff ${d_ff} \
  --target '日前统一结算价' \
  --date_record \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /data/qiuyunzhong/DATA/Xiexin/ \
  --data_path Actual-Price-riqian.csv \
  --model_id $data_$seq_len_$pred_len \
  --model $model_name \
  --data $data \
  --features MS \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 1 \
  --factor 3 \
  --enc_in 11 \
  --dec_in 11 \
  --c_out 11 \
  --des $des \
  --batch_size 32 \
  --learning_rate 5e-6 \
  --lradj cosine \
  --patience 5 \
  --train_epochs 30 \
  --d_model ${d_model} \
  --d_ff ${d_ff} \
  --target '日前统一结算价' \
  --date_record \
  --inverse \
  --itr 1