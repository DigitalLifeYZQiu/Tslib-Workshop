# 定义日志文件
LOG_FILE="scriptsTimeBert/pt/TimeBert.log"

# 定义要执行的脚本列表（按顺序执行）
SCRIPTS=(
    "scriptsTimeBert/pt/TimeBert_MSL.sh"
    "scriptsTimeBert/pt/TimeBert_PSM.sh"
    "scriptsTimeBert/pt/TimeBert_SMAP.sh"
    "scriptsTimeBert/pt/TimeBert_SMD.sh"
    "scriptsTimeBert/pt/TimeBert_SWAT.sh"
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