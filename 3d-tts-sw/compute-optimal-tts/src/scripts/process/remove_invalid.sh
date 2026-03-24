#!/bin/bash
# 删除不在配置列表中的模型组合
# 
# 用法:
#   ./remove_invalid.sh           # 只检查，不删除
#   ./remove_invalid.sh --delete  # 检查并可选择删除

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/remove_invalid_combinations.py"

# 检查 Python 脚本是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误: 找不到 Python 脚本: $PYTHON_SCRIPT"
    exit 1
fi

# 运行 Python 脚本，传递所有参数
python3 "$PYTHON_SCRIPT" "$@"
