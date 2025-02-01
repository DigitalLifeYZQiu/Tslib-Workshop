#export CUDA_VISIBLE_DEVICES=0,1,2,3  # 所有可用 GPU
#
#ckpt_path="/data/zhanghaoran/projects/ltsm/experiments/bert/bert/lotsa_uea_bert_d256_l4_p4_n512_weight_p25_wo_revin_long/20012025_231911/ckpt/bert/bert/lotsa_uea_bert_d256_l4_p4_n512_weight_p25_wo_revin_long-epoch=01.ckpt"
##ckpt_path="ckpt_path=/data/zhanghaoran/projects/ltsm/experiments/bert/bert/uea_bert_d1024_l8_p96_n512_weight_p50_wo_revin_full/03012025_044649/ckpt/bert/bert/uea_bert_d1024_l8_p96_n512_weight_p50_wo_revin_full-epoch=06.ckpt"
## 参数组合列表（共 1*2*6*5=60 种组合）
#knn_neighbors_list=(1 2 3 4 5 6 7 8 9 10)
#model_name=TimeBert
#lr=3e-5
#lradj=cosine
#
## 最大并行任务数（根据 GPU 显存调整）
#max_jobs=4
#job_counter=0  # 任务计数器（用于 GPU 轮询分配）
#
#mkdir -p logs/TimeBert_PEMS  # 创建日志目录
#
## 遍历所有参数组合
#for knn_neighbors in "${knn_neighbors_list[@]}"; do
#
#  # 动态分配 GPU（任务计数器对 GPU 数量取模）
#  gpu_id=$((job_counter % 4))
#
#  # 打印任务信息
#  echo "启动任务：seq_len=$seq_len mask_rate=$mask_rate lr=$lr lradj=$lradj epoch=$epoch → GPU $gpu_id"
#
#  # 后台运行任务（关键：用子 shell 隔离变量）
#  (
#    export CUDA_VISIBLE_DEVICES=$gpu_id
#    python -u run.py \
#      --task_name classification_ablation \
#      --is_training 0 \
#      --root_path ./dataset/PEMS-SF/ \
#      --model_id PEMS-SF \
#      --model $model_name \
#      --data UEA \
#      --e_layers 3 \
#      --batch_size 1 \
#      --d_model 128 \
#      --d_ff 256 \
#      --top_k 3 \
#      --des 'Exp' \
#      --itr 1 \
#      --learning_rate $lr \
#      --ckpt_path $ckpt_path \
#      --train_epochs 50 \
#      --patience 10 \
#      --lradj $lradj \
#      --date_record \
#      --knn_neighbors $knn_neighbors \
#      > "logs/TimeBert_PEMS/TimeBert_PEMS_knn${knn_neighbors}.log" 2>&1
#  ) &
#
#  # 更新任务计数器
#  ((job_counter++))
#
#  # 控制并行度：如果后台任务数 ≥ max_jobs，等待
#  while [[ $(jobs -r | wc -l) -ge $max_jobs ]]; do
#    sleep 5
#  done
#done
#
#wait  # 等待所有后台任务完成
#echo "所有任务已完成！日志保存在 logs/ 目录"

export CUDA_VISIBLE_DEVICES=0

model_name=TimeBert

# 第一个参数为学习率
lr=3e-5
ckpt_path=/data/zhanghaoran/projects/ltsm/experiments/bert/bert/lotsa_uea_bert_d768_l12_p36_n512_weight_p25_wo_revin_long/08012025_010843/ckpt/bert/bert/lotsa_uea_bert_d768_l12_p36_n512_weight_p25_wo_revin_long-epoch=02.ckpt
lradj=cosine

for knn_neighbors in 1 2 3 4 5 6 7 8 9 10
do

echo knn_neighbors $knn_neighbors
python -u run.py \
  --task_name classification_ablation \
  --is_training 0 \
  --root_path ./dataset/PEMS-SF/ \
  --model_id PEMS-SF \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 4 \
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
  --knn_neighbors $knn_neighbors
#  --cls_mask_token_only

echo knn_neighbors $knn_neighbors
python -u run.py \
  --task_name classification_ablation \
  --is_training 0 \
  --root_path ./dataset/PEMS-SF/ \
  --model_id PEMS-SF \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 4 \
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
  --knn_neighbors $knn_neighbors \
  --cls_mask_token_only

done

# > scriptsTimeBert/classification/TimeBert-PEMS.log && bash scriptsTimeBert/classification/TimeBert-PEMS.sh 2>&1 | tee -a scriptsTimeBert/classification/TimeBert-PEMS.log