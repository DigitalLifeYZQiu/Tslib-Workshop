export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

model_name=TimeBert

# 第一个参数为学习率
lr=3e-5
ckpt_path=/data/zhanghaoran/projects/ltsm/experiments/bert/bert/lotsa_uea_bert_d768_l12_p36_n512_weight_p25_wo_revin_long/08012025_010843/ckpt/bert/bert/lotsa_uea_bert_d768_l12_p36_n512_weight_p25_wo_revin_long-epoch=02.ckpt
lradj=cosine
knn_neighbors_list=(1)
dataset_dir="/data/zhanghaoran/projects/iTransformer_exp/dataset/UCR"

# 最大并行任务数（根据 GPU 显存调整）
max_jobs=4
job_counter=0  # 任务计数器（用于 GPU 轮询分配）

mkdir -p logs/TimeBert_UCR  # 创建日志目录

# ergodic datasets
for knn_neighbors in "${knn_neighbors_list[@]}"; do
for file_path in "$dataset_dir"/*; do
  data=$(basename "$file_path")
  # 动态分配 GPU（任务计数器对 GPU 数量取模）
  gpu_id=$((job_counter % 8))
  # 打印任务信息
  echo "启动任务：data=$data → GPU $gpu_id"
  (
    python -u run.py \
    --task_name classification_ablation \
    --is_training 0 \
    --root_path /data/zhanghaoran/projects/iTransformer_exp/dataset/UCR/$data \
    --model_id $data \
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
    --lradj $lradj \
    --patience 10 \
    --date_record \
    --knn_neighbors $knn_neighbors \
    > "logs/TimeBert_UCR/TimeBert_UCR_${data}_knn${knn_neighbors}.log" 2>&1
  ) &

  # 更新任务计数器
  ((job_counter++))

  # 控制并行度：如果后台任务数 ≥ max_jobs，等待
  while [[ $(jobs -r | wc -l) -ge $max_jobs ]]; do
    sleep 5
  done
done
done

# > scriptsTimeBert/classification/TimeBert.log && bash scriptsTimeBert/classification/TimeBert.sh 2>&1 | tee -a scriptsTimeBert/classification/TimeBert.log