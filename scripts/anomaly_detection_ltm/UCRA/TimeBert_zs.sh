export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 所有可用 GPU

# 定义每个脚本使用的显卡ID
GPUS=(0 1 2 3 4 5 6 7)
NumGPUS=${#GPUS[@]}
# 最大并行任务数（根据 GPU 显存调整）
max_jobs=8
job_counter=0  # 任务计数器（用于 GPU 轮询分配）



dataset_dir="../DATA/UCR_Anomaly_FullData"
ckpt_path="/data/zhanghaoran/projects/ltsm/experiments/bert/bert/lotsa_uea_bert_d256_l4_p4_n512_weight_p25_wo_revin_long/20012025_231911/ckpt/bert/bert/lotsa_uea_bert_d256_l4_p4_n512_weight_p25_wo_revin_long-epoch=01.ckpt"


log_dir=logs/UCRA_ltm/TimeBERT_zs/logger
mkdir -p ${log_dir}  # 创建日志目录
csv_dir=logs/UCRA_ltm/TimeBERT_zs/results
mkdir -p ${csv_dir}  # 创建结果目录

for file_path in "$dataset_dir"/*
do
  data=$(basename "$file_path")
  gpu="${GPUS[$((job_counter % NumGPUS))]}"

  # 生成对应的日志文件名
  log_file="${data}.log"
  # 清空日志文件
  > "${log_dir}/${log_file}"

  echo "========== Running $data mask_rate=${mask_rate} on GPU $gpu =========="
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    python -u run.py \
      --task_name anomaly_detection_ltm \
      --is_training 0 \
      --ckpt_path $ckpt_path \
      --root_path  ${dataset_dir} \
      --data_path $data \
      --model_id UCRA_$data \
      --model TimeBert \
      --data UCRA \
      --features M \
      --seq_len 512 \
      --stride 1 \
      --pred_len 0 \
      --batch_size 64 \
      --train_epochs 10 \
      --csv_file "${csv_dir}/results.csv" \
      --date_record \
      > "${log_dir}/${log_file}" 2>&1
  ) &
   # 更新任务计数器
  ((job_counter++))

  # 控制并行度：如果后台任务数 ≥ max_jobs，等待
  while [[ $(jobs -r | wc -l) -ge $max_jobs ]]; do
    sleep 5
  done
done

# 等待所有后台任务完成
wait

echo "✅ All scripts finished. Each script's log is saved in its respective .log file."

#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 所有可用 GPU
#
## 定义每个脚本使用的显卡ID
#GPUS=(0 1 2 3 4 5 6 7)
#NumGPUS=${#GPUS[@]}
## 最大并行任务数（根据 GPU 显存调整）
#max_jobs=8
#job_counter=0  # 任务计数器（用于 GPU 轮询分配）
#
#
#
#dataset_dir="../DATA/UCR_Anomaly_FullData"
#ckpt_path="/data/zhanghaoran/projects/ltsm/experiments/bert/bert/lotsa_uea_bert_d256_l4_p4_n512_weight_p25_wo_revin_long/20012025_231911/ckpt/bert/bert/lotsa_uea_bert_d256_l4_p4_n512_weight_p25_wo_revin_long-epoch=01.ckpt"
#
#for mask_rate in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
#do
#  log_dir=logs/UCRA_ltm/TimeBERT_zs_mask${mask_rate}/logger
#  mkdir -p ${log_dir}  # 创建日志目录
#  csv_dir=logs/UCRA_ltm/TimeBERT_zs_mask${mask_rate}/results
#  mkdir -p ${csv_dir}  # 创建结果目录
#
#  for file_path in "$dataset_dir"/*
#  do
#    data=$(basename "$file_path")
#    gpu="${GPUS[$((job_counter % NumGPUS))]}"
#
#    # 生成对应的日志文件名
#    log_file="${data}.log"
#    # 清空日志文件
#    > "${log_dir}/${log_file}"
#
#    echo "========== Running $data mask_rate=${mask_rate} on GPU $gpu =========="
#    (
#      export CUDA_VISIBLE_DEVICES=$gpu
#      python -u run.py \
#        --task_name anomaly_detection_ltm \
#        --is_training 0 \
#        --ckpt_path $ckpt_path \
#        --root_path  ${dataset_dir} \
#        --data_path $data \
#        --model_id UCRA_$data \
#        --model TimeBert \
#        --data UCRA \
#        --features M \
#        --seq_len 512 \
#        --stride 1 \
#        --pred_len 0 \
#        --batch_size 64 \
#        --train_epochs 10 \
#        --csv_file "${csv_dir}/results.csv" \
#        --date_record \
#        --ad_mask_type random \
#        --mask_rate ${mask_rate} \
#        > "${log_dir}/${log_file}" 2>&1
#    ) &
#     # 更新任务计数器
#    ((job_counter++))
#
#    # 控制并行度：如果后台任务数 ≥ max_jobs，等待
#    while [[ $(jobs -r | wc -l) -ge $max_jobs ]]; do
#      sleep 5
#    done
#  done
#done
#
## 等待所有后台任务完成
#wait
#
#echo "✅ All scripts finished. Each script's log is saved in its respective .log file."