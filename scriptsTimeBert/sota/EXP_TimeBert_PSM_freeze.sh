export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 所有可用 GPU

ckpt_path="/data/zhanghaoran/projects/ltsm/experiments/bert/bert/lotsa_uea_bert_d256_l4_p4_n512_weight_p25_wo_revin_long/20012025_231911/ckpt/bert/bert/lotsa_uea_bert_d256_l4_p4_n512_weight_p25_wo_revin_long-epoch=01.ckpt"

# 参数组合列表（共 1*2*6*5=60 种组合）
seq_len_list=(40)
mask_rate_list=(0 0.25 0.5 0.75)
lr_list=(0.001 0.00075 0.0005 0.00025 0.0001 0.000075 0.00005 0.000025 0.00001)
lradj_list=(cosine type1)
epoch_list=(3 5 10 15 20)

# 最大并行任务数（根据 GPU 显存调整）
max_jobs=1
job_counter=0  # 任务计数器（用于 GPU 轮询分配）

mkdir -p logs/TimeBert_PSM_freeze  # 创建日志目录

# 遍历所有参数组合
for seq_len in "${seq_len_list[@]}"; do
  for mask_rate in "${mask_rate_list[@]}"; do
    for lr in "${lr_list[@]}"; do
      for lradj in "${lradj_list[@]}"; do
        for epoch in "${epoch_list[@]}"; do

          # 动态分配 GPU（任务计数器对 GPU 数量取模）
          gpu_id=$((job_counter % 8))

          # 打印任务信息
          echo "启动任务：seq_len=$seq_len mask_rate=$mask_rate lr=$lr lradj=$lradj epoch=$epoch → GPU $gpu_id"

          # 后台运行任务（关键：用子 shell 隔离变量）
          (
#            export CUDA_VISIBLE_DEVICES=$gpu_id
            export CUDA_VISIBLE_DEVICES=1
            python -u run.py \
              --task_name anomaly_detection \
              --is_training 1 \
              --ckpt_path $ckpt_path \
              --root_path ./dataset/PSM \
              --model_id PSM \
              --model TimeBert \
              --data PSM \
              --features M \
              --seq_len $seq_len \
              --pred_len 0 \
              --batch_size 16 \
              --patch_len 4 \
              --train_epochs $epoch \
              --stride 40 \
              --enc_in 25 \
              --anomaly_ratio 1 \
              --learning_rate $lr \
              --ad_mask_type random \
              --mask_rate $mask_rate \
              --date_record \
              --freeze_patch_encoder \
              --lradj "$lradj" \
              > "logs/TimeBert_PSM_freeze/TimeBert_PSM_freeze_seq${seq_len}_mask${mask_rate}_lr${lr}-${lradj}_epoch${epoch}.log" 2>&1
          ) &

          # 更新任务计数器
          ((job_counter++))

          # 控制并行度：如果后台任务数 ≥ max_jobs，等待
          while [[ $(jobs -r | wc -l) -ge $max_jobs ]]; do
            sleep 5
          done
        done
      done
    done
  done
done

wait  # 等待所有后台任务完成
echo "所有任务已完成！日志保存在 logs/ 目录"


#export CUDA_VISIBLE_DEVICES=4
#
##ckpt_path="/data/zhanghaoran/projects/ltsm/experiments/bert/bert/lotsa_uea_bert_d768_l12_p36_n512_weight_p25_wo_revin_full/04012025_193158/ckpt/bert/bert/lotsa_uea_bert_d768_l12_p36_n512_weight_p25_wo_revin_full-epoch=14.ckpt"
##ckpt_path="/data/zhanghaoran/projects/ltsm/experiments/bert/bert/lotsa_uea_bert_d768_l12_p4_n512_weight_p50_wo_revin_full/07012025_015048/ckpt/bert/bert/lotsa_uea_bert_d768_l12_p4_n512_weight_p50_wo_revin_full-epoch=11.ckpt"
#ckpt_path="/data/zhanghaoran/projects/ltsm/experiments/bert/bert/lotsa_uea_bert_d256_l4_p4_n512_weight_p25_wo_revin_long/20012025_231911/ckpt/bert/bert/lotsa_uea_bert_d256_l4_p4_n512_weight_p25_wo_revin_long-epoch=01.ckpt"
#for seq_len in 40
#do
#for mask_rate in 0.1
#do
#echo mask rate $mask_rate
#python -u run.py \
#  --task_name anomaly_detection \
#  --is_training 1 \
#  --ckpt_path $ckpt_path \
#  --root_path ./dataset/PSM \
#  --model_id PSM \
#  --model TimeBert \
#  --data PSM \
#  --features M \
#  --seq_len $seq_len \
#  --pred_len 0 \
#  --batch_size 16 \
#  --patch_len 4 \
#  --train_epochs 2 \
#  --stride 40 \
#  --enc_in 25 \
#  --anomaly_ratio 1 \
#  --learning_rate 0.0001 \
#  --ad_mask_type random \
#  --mask_rate $mask_rate \
#  --date_record
#done
#done
#  > scriptsTimeBert/fullft-1/TimeBert_PSM.log && bash scriptsTimeBert/fullft-1/TimeBert_PSM.sh 2>&1 | tee -a scriptsTimeBert/fullft-1/TimeBert_PSM.log