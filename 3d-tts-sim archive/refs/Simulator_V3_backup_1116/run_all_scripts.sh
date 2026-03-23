#!/bin/bash

# 脚本功能：执行当前目录下所有以main_for_开头的Python文件

echo "开始执行所有以main_for_开头的Python脚本..."
echo "----------------------------------------"

# 查找所有以main_for_开头的Python文件
files=(main_for_*.py)

# 检查是否有符合条件的文件
if [ ${#files[@]} -eq 0 ]; then
    echo "未找到以main_for_开头的Python文件"
    exit 0
fi

echo "找到 ${#files[@]} 个文件需要执行："
for file in "${files[@]}"; do
    echo "- $file"
done
echo "----------------------------------------"

# 逐个执行文件
for file in "${files[@]}"; do
    echo "正在执行：$file"
    echo "===================================="
    
    # 执行Python文件
    python3 "$file"
    
    # 记录执行状态
    status=$?
    
    if [ $status -eq 0 ]; then
        echo "✅ $file 执行成功"
    else
        echo "❌ $file 执行失败，退出码：$status"
    fi
    
    echo "===================================="
done

echo "----------------------------------------"
echo "所有脚本执行完毕！"