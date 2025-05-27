export CUDA_VISIBLE_DEVICES=3

model_name=DLinear
data=Xiexin_PowerLoad
OT=sum
root_path=/data/qiuyunzhong/AINode-LTM-HF/Xiexindata/output_sn_plots

mkdir -p logs/${model_name}/
logger=logs/${model_name}/${model_name}_${data}.log
> "${logger}"

# ETTh1
for seq_len in 160
do
for pred_len in 160
do
for id in 3205001139 3205008027 3205008043 3205003017 3300002014 3205000409 3205004059 3205004069
do
data_path=${id}_sum.csv

echo ${data_path}
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id $data_$seq_len_$pred_len \
  --model $model_name \
  --data $data \
  --features S \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --des 'Exp' \
  --target $OT \
  --inverse \
  --batch_size 1024 \
  --e_layers 1 \
  --date_record \
  --itr 1

echo ${data_path}
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id $data_$seq_len_$pred_len \
  --model $model_name \
  --data $data \
  --features S \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --des 'Exp' \
  --target $OT \
  --batch_size 1024 \
  --e_layers 1 \
  --date_record \
  --itr 1

echo ${data_path}
done
done
done 2>&1 | tee -a "${logger}"