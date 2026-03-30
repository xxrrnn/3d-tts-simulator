#!/bin/bash
# 20260330
# ============================================================================
# 自动完成评估脚本
# 功能：循环执行评估，检查并删除不完整的结果，直到所有评估都完成
# ============================================================================

# 颜色定义
RED='\033[91m'
GREEN='\033[92m'
YELLOW='\033[93m'
BLUE='\033[94m'
CYAN='\033[96m'
BOLD='\033[1m'
END='\033[0m'

# 脚本路径配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_all_combinations.sh"
CHECK_SCRIPT="${SCRIPT_DIR}/process/check_incomplete_questions.py"
SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${SRC_DIR}/output"

# 最大重试次数
MAX_ITERATIONS=10

# 日志函数
log_message() {
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo -e "$message"
}

log_section() {
    local message="$1"
    echo -e "\n${BOLD}${CYAN}========================================${END}"
    echo -e "${BOLD}${CYAN}$message${END}"
    echo -e "${BOLD}${CYAN}========================================${END}\n"
}

# 检查脚本是否存在
check_scripts() {
    log_message "检查必需的脚本文件..."
    
    if [ ! -f "$EVAL_SCRIPT" ]; then
        log_message "${RED}错误: 评估脚本不存在: $EVAL_SCRIPT${END}"
        exit 1
    fi
    
    if [ ! -f "$CHECK_SCRIPT" ]; then
        log_message "${RED}错误: 检查脚本不存在: $CHECK_SCRIPT${END}"
        exit 1
    fi
    
    log_message "${GREEN}✓ 所有脚本文件存在${END}"
}

# 执行评估脚本
run_evaluation() {
    local iteration=$1
    log_section "第 ${iteration} 次评估 - 开始"
    log_message "执行: $EVAL_SCRIPT"
    
    # 切换到 src 目录执行评估脚本（脚本需要在 src 目录运行）
    if (cd "$SRC_DIR" && bash "$EVAL_SCRIPT"); then
        log_message "${GREEN}✓ 评估脚本执行完成${END}"
        return 0
    else
        local exit_code=$?
        log_message "${RED}✗ 评估脚本执行失败，退出码: $exit_code${END}"
        return $exit_code
    fi
}

# 检查不完整的问题
check_incomplete() {
    log_section "检查不完整的结果"
    log_message "执行: $CHECK_SCRIPT --output-dir $OUTPUT_DIR"
    
    # 运行检查脚本，捕获输出
    local check_output=$(python3 "$CHECK_SCRIPT" --output-dir "$OUTPUT_DIR" 2>&1)
    local exit_code=$?
    
    # 显示输出
    echo "$check_output"
    
    # 检查输出中是否包含"所有配置都正常"
    if echo "$check_output" | grep -q "所有配置都正常"; then
        log_message "${GREEN}✓ 所有配置都正常！${END}"
        return 0
    else
        log_message "${YELLOW}⚠ 发现不完整的配置${END}"
        return 1
    fi
}

# 删除不完整的结果
delete_incomplete() {
    log_section "删除不完整的结果"
    log_message "执行: $CHECK_SCRIPT --delete --output-dir $OUTPUT_DIR"
    
    # 使用 expect 自动化交互，选择删除所有有问题的配置（选项3）
    # 如果没有 expect，使用 python 脚本的自动确认方式
    if command -v expect &> /dev/null; then
        expect << EOF
set timeout -1
spawn python3 "$CHECK_SCRIPT" --delete --output-dir "$OUTPUT_DIR"
expect {
    "请选择 *:" {
        send "3\r"
        exp_continue
    }
    "确认删除*:" {
        send "yes\r"
        exp_continue
    }
    eof
}
EOF
    else
        # 如果没有 expect，尝试使用 yes 命令自动回答
        log_message "${YELLOW}注意: 系统未安装 expect，尝试使用备用方法${END}"
        echo -e "3\nyes" | python3 "$CHECK_SCRIPT" --delete --output-dir "$OUTPUT_DIR"
    fi
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        log_message "${GREEN}✓ 删除操作完成${END}"
    else
        log_message "${YELLOW}⚠ 删除操作可能未完成，退出码: $exit_code${END}"
    fi
    
    # 等待一下，确保文件系统同步
    sleep 5
    
    return $exit_code
}

# 主函数
main() {
    log_section "自动完成评估脚本启动"
    log_message "评估脚本: $EVAL_SCRIPT"
    log_message "检查脚本: $CHECK_SCRIPT"
    log_message "输出目录: $OUTPUT_DIR"
    log_message "最大迭代次数: $MAX_ITERATIONS"
    
    # 检查脚本文件
    check_scripts
    
    # 迭代计数
    local iteration=1
    
    while [ $iteration -le $MAX_ITERATIONS ]; do
        log_section "第 ${iteration}/${MAX_ITERATIONS} 轮迭代"
        
        # 1. 执行评估
        if ! run_evaluation $iteration; then
            log_message "${RED}✗ 评估失败，停止迭代${END}"
            exit 1
        fi
        
        # 2. 检查是否有不完整的结果
        if check_incomplete; then
            log_section "完成！"
            log_message "${GREEN}${BOLD}✓ 所有评估都已完成！${END}"
            log_message "总迭代次数: $iteration"
            exit 0
        fi
        
        # 3. 删除不完整的结果
        delete_incomplete
        
        # 4. 增加迭代计数
        iteration=$((iteration + 1))
        
        # 如果还没有达到最大次数，继续下一轮
        if [ $iteration -le $MAX_ITERATIONS ]; then
            log_message "${CYAN}准备开始第 ${iteration} 轮迭代...${END}"
            sleep 3
        fi
    done
    
    # 如果达到最大迭代次数仍未完成
    log_section "警告"
    log_message "${RED}${BOLD}⚠ 已达到最大迭代次数 ($MAX_ITERATIONS)，但仍有未完成的评估${END}"
    log_message "请手动运行检查脚本: python3 $CHECK_SCRIPT --output-dir $OUTPUT_DIR"
    exit 1
}

# 捕获 Ctrl+C 信号
trap 'log_message "${YELLOW}接收到中断信号，正在退出...${END}"; exit 130' INT TERM

# 运行主函数
main
