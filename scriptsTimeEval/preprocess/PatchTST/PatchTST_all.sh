# 定义日志文件
LOG_FILE="scriptsTimeEval/preprocess/PatchTST/PatchTST.log"

# 定义要执行的脚本列表（按顺序执行）
SCRIPTS=(
#    "scriptsTimeEval/preprocess/PatchTST/PatchTST_ETTh1.sh"
#    "scriptsTimeEval/preprocess/PatchTST/PatchTST_ETTh2.sh"
#    "scriptsTimeEval/preprocess/PatchTST/PatchTST_ETTm1.sh"
#    "scriptsTimeEval/preprocess/PatchTST/PatchTST_ETTm2.sh"
#    "scriptsTimeEval/preprocess/PatchTST/PatchTST_ECL.sh"
    "scriptsTimeEval/preprocess/PatchTST/PatchTST_Traffic.sh"
    "scriptsTimeEval/preprocess/PatchTST/PatchTST_Weather.sh"
    "scriptsTimeEval/preprocess/PatchTST/PatchTST_Exchange.sh"
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