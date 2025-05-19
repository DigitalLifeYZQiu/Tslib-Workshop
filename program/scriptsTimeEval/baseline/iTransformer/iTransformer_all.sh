#bash scriptsTimeEval/baseline/iTransformer/iTransformer_ETTh1.sh
#bash scriptsTimeEval/baseline/iTransformer/iTransformer_ETTh2.sh
#bash scriptsTimeEval/baseline/iTransformer/iTransformer_ETTm1.sh
#bash scriptsTimeEval/baseline/iTransformer/iTransformer_ETTm2.sh
#bash scriptsTimeEval/baseline/iTransformer/iTransformer_ECL.sh
#bash scriptsTimeEval/baseline/iTransformer/iTransformer_Traffic.sh
#bash scriptsTimeEval/baseline/iTransformer/iTransformer_Weather.sh
#bash scriptsTimeEval/baseline/iTransformer/iTransformer_Exchange.sh

# 定义日志文件
LOG_FILE="scriptsTimeEval/baseline/iTransformer/iTransformer.log"

# 定义要执行的脚本列表（按顺序执行）
SCRIPTS=(
#    "scriptsTimeEval/baseline/iTransformer/iTransformer_ETTh1.sh"
#    "scriptsTimeEval/baseline/iTransformer/iTransformer_ETTh2.sh"
#    "scriptsTimeEval/baseline/iTransformer/iTransformer_ETTm1.sh"
#    "scriptsTimeEval/baseline/iTransformer/iTransformer_ETTm2.sh"
#    "scriptsTimeEval/baseline/iTransformer/iTransformer_ECL.sh"
    "scriptsTimeEval/baseline/iTransformer/iTransformer_Traffic.sh"
    "scriptsTimeEval/baseline/iTransformer/iTransformer_Weather.sh"
    "scriptsTimeEval/baseline/iTransformer/iTransformer_Exchange.sh"
)

# 清空日志文件
> "$LOG_FILE"

# 遍历并执行每个脚本
for script in "${SCRIPTS[@]}"; do
    if [[ -x "$script" ]]; then  # 判断脚本是否有执行权限
        echo "========== Running $script ==========" | tee -a "$LOG_FILE"
        bash "$script" 2>&1 | tee -a "$LOG_FILE"
        echo "========== Finished $script ==========" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
    else
        echo "⚠️  $script is not executable or does not exist." | tee -a "$LOG_FILE"
    fi
done

echo "✅ All scripts finished. Logs are saved in $LOG_FILE."