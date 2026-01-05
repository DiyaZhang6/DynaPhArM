#!/bin/bash


export PYTHONPATH="/home/zdy/FlexPose:$PYTHONPATH"

# 固定不变的蛋白质文件路径
PROTEIN_FILE="/home/zdy/Project2/data/case/HIV-1-Protease/1hpv_clean.pdb"

# 存放所有 MOL2 配体文件的文件夹路径
LIGAND_DIR="/home/zdy/Project2/data/case/HIV-1-Protease/mol2/"

# 存放 Python 脚本 'demo.py' 的路径
PYTHON_SCRIPT="/home/zdy/FlexPose/demo.py"

# 输出结果的文件夹
OUTPUT_DIR_BASE="batch_output"

# 确保基础输出目录存在
mkdir -p "$OUTPUT_DIR_BASE"

# --- 2. 主循环 ---

# 检查配体目录是否存在
if [ ! -d "$LIGAND_DIR" ]; then
    echo "错误：配体文件夹 '$LIGAND_DIR' 不存在。"
    exit 1
fi

echo "--- 开始批量运行 FlexPose ---"
echo "蛋白质: $PROTEIN_FILE"
echo "配体文件夹: $LIGAND_DIR"
echo "---------------------------------"

# 使用 find 和 while read 循环来处理所有 .mol2 文件
# 这种方法能很好地处理包含空格等特殊字符的文件名
find "$LIGAND_DIR" -name "*.mol2" | while read ligand_path; do
    
    # 从完整路径中提取配体的基本文件名 (例如 "ZINC000822662146")
    ligand_name=$(basename "$ligand_path" .mol2)
    
    echo "正在处理: $ligand_name"

    # 为当前配体创建一个独立的输出文件夹
    current_output_dir="$OUTPUT_DIR_BASE/$ligand_name"
    mkdir -p "$current_output_dir"

    # 为当前配体定义独立的输出文件名
    output_csv_path="$current_output_dir/output.csv"
    structure_output_path="$current_output_dir/structure_output/"

    # --- 3. 修改并运行 Python 脚本 ---
    
    # 动态生成一个临时的 Python 脚本，填入正确的路径
    # 使用 "heredoc" 语法 (<<EOF ... EOF) 来写入多行文本
    cat > temp_run.py <<EOF
from FlexPose.utils.prediction import predict as predict_by_FlexPose

predict_by_FlexPose(
    protein="$PROTEIN_FILE",
    ligand="$ligand_path",
    ref_pocket_center="$ligand_path",
    device='cuda:0',
    structure_output_path="$structure_output_path",
    output_result_path="$output_csv_path"
)
EOF

    # 运行这个临时的 Python 脚本
    # 确保您激活了正确的 Conda 环境！
    python temp_run.py
    
    echo "完成: $ligand_name. 结果保存在 $current_output_dir"
    echo "---------------------------------"

done

# 清理临时的 Python 脚本
rm temp_run.py

echo "--- 批量处理完成 ---"