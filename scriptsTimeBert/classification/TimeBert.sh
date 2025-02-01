export CUDA_VISIBLE_DEVICES=1

model_name=TimeBert

# 第一个参数为学习率
lr=3e-5
ckpt_path=/data/zhanghaoran/projects/ltsm/experiments/bert/bert/lotsa_uea_bert_d768_l12_p36_n512_weight_p25_wo_revin_long/08012025_010843/ckpt/bert/bert/lotsa_uea_bert_d768_l12_p36_n512_weight_p25_wo_revin_long-epoch=02.ckpt
lradj=cosine

for knn_neighbors in 1
do
echo knn_neighbors $knn_neighbors
python -u run.py \
  --task_name classification_ablation \
  --is_training 0 \
  --root_path ./dataset/EthanolConcentration/ \
  --model_id EthanolConcentration \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate $lr \
  --ckpt_path $ckpt_path \
  --train_epochs 50 \
  --patience 10 \
  --lradj $lradj \
  --date_record \
  --knn_neighbors $knn_neighbors
#  --cls_mask_token_only

echo knn_neighbors $knn_neighbors
python -u run.py \
  --task_name classification_ablation \
  --is_training 0 \
  --root_path ./dataset/FaceDetection/ \
  --model_id FaceDetection \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 8 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate $lr \
  --ckpt_path $ckpt_path \
  --train_epochs 50 \
  --patience 10 \
  --lradj $lradj \
  --date_record \
  --knn_neighbors $knn_neighbors
#  --cls_mask_token_only

echo knn_neighbors $knn_neighbors
python -u run.py \
  --task_name classification_ablation \
  --is_training 0 \
  --root_path ./dataset/Handwriting/ \
  --model_id Handwriting \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate $lr \
  --ckpt_path $ckpt_path \
  --train_epochs 50 \
  --patience 10 \
  --lradj $lradj \
  --date_record \
  --knn_neighbors $knn_neighbors
#  --cls_mask_token_only

echo knn_neighbors $knn_neighbors
python -u run.py \
  --task_name classification_ablation \
  --is_training 0 \
  --root_path ./dataset/Heartbeat/ \
  --model_id Heartbeat \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate $lr \
  --ckpt_path $ckpt_path \
  --train_epochs 50 \
  --patience 10 \
  --lradj $lradj \
  --date_record \
  --knn_neighbors $knn_neighbors
#  --cls_mask_token_only

echo knn_neighbors $knn_neighbors
python -u run.py \
  --task_name classification_ablation \
  --is_training 0 \
  --root_path ./dataset/JapaneseVowels/ \
  --model_id JapaneseVowels \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate $lr \
  --ckpt_path $ckpt_path \
  --train_epochs 50 \
  --patience 10 \
  --lradj $lradj \
  --date_record \
  --knn_neighbors $knn_neighbors
#  --cls_mask_token_only

#echo knn_neighbors $knn_neighbors
#python -u run.py \
#  --task_name classification_ablation \
#  --is_training 1 \
#  --root_path ./dataset/PEMS-SF/ \
#  --model_id PEMS-SF \
#  --model $model_name \
#  --data UEA \
#  --e_layers 3 \
#  --batch_size $batch_size \
#  --d_model 128 \
#  --d_ff 256 \
#  --top_k 3 \
#  --des 'Exp' \
#  --itr 1 \
#  --learning_rate $lr \
#  --ckpt_path $ckpt_path \
#  --train_epochs 50 \
#  --patience 10 \
#  --lradj $lradj \
#  --date_record \
#  --knn_neighbors $knn_neighbors
##  --cls_mask_token_only

echo knn_neighbors $knn_neighbors
python -u run.py \
  --task_name classification_ablation \
  --is_training 0 \
  --root_path ./dataset/SelfRegulationSCP1/ \
  --model_id SelfRegulationSCP1 \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate $lr \
  --ckpt_path $ckpt_path \
  --train_epochs 50 \
  --patience 10 \
  --lradj $lradj \
  --date_record \
  --knn_neighbors $knn_neighbors
#  --cls_mask_token_only

echo knn_neighbors $knn_neighbors
python -u run.py \
  --task_name classification_ablation \
  --is_training 0 \
  --root_path ./dataset/SelfRegulationSCP2/ \
  --model_id SelfRegulationSCP2 \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate $lr \
  --ckpt_path $ckpt_path \
  --train_epochs 50 \
  --patience 10 \
  --lradj $lradj \
  --date_record \
  --knn_neighbors $knn_neighbors
#  --cls_mask_token_only

echo knn_neighbors $knn_neighbors
python -u run.py \
  --task_name classification_ablation \
  --is_training 0 \
  --root_path ./dataset/SpokenArabicDigits/ \
  --model_id SpokenArabicDigits \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate $lr \
  --ckpt_path $ckpt_path \
  --train_epochs 50 \
  --patience 10 \
  --lradj $lradj \
  --date_record \
  --knn_neighbors $knn_neighbors
#  --cls_mask_token_only

echo knn_neighbors $knn_neighbors
python -u run.py \
  --task_name classification_ablation \
  --is_training 0 \
  --root_path ./dataset/UWaveGestureLibrary/ \
  --model_id UWaveGestureLibrary \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate $lr \
  --ckpt_path $ckpt_path \
  --train_epochs 50 \
  --patience 10 \
  --lradj $lradj \
  --date_record \
  --knn_neighbors $knn_neighbors
#  --cls_mask_token_only
done

# > scriptsTimeBert/classification/TimeBert.log && bash scriptsTimeBert/classification/TimeBert.sh 2>&1 | tee -a scriptsTimeBert/classification/TimeBert.log