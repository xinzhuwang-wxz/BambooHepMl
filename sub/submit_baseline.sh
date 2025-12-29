#!/bin/bash
###### Part 1: SLURM 配置 ######
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --account=higgsgpu
#SBATCH --job-name=baseline_train
#SBATCH --ntasks=1
#SBATCH --output=logs/baseline_%j.log
#SBATCH --error=logs/baseline_%j.err
#SBATCH --mem-per-cpu=24576
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00

###### Part 2: 训练配置（在这里修改参数）######
# 所有训练参数都在这里配置，修改后直接运行 sbatch sub/submit_baseline.sh 即可

# ========== 数据配置 ==========
data_config_path="configs/data_example.yaml"  # 数据配置文件路径（data.yaml，如果使用字典方式可留空）
data_treename="tree"                          # ROOT 树名

# 数据源路径（支持两种方式，类似 weaver）
# 方式 1: 字典方式（推荐，用于多分类）- 自动生成 one-hot 编码标签
#   格式：label:path，例如：data_train="a:/path/to/a/*.root b:/path/to/b/*.root c:/path/to/c/*.root"
#   系统会自动生成 is_a, is_b, is_c 作为 one-hot 编码标签，无需手动配置 data.yaml 中的 labels
# 方式 2: 混合方式 - 需要在 data.yaml 中配置 labels
#   格式：直接路径，例如：data_train="/path/to/mixed/*.root"
data_train="a:/path/to/a/*.root b:/path/to/b/*.root c:/path/to/c/*.root"  # 训练数据
data_val="a:/path/to/val/a/*.root b:/path/to/val/b/*.root c:/path/to/val/c/*.root"  # 验证数据（可选，留空则从训练数据分割）
data_test=""  # 测试数据（可选，可以是字典方式或混合方式）

# ========== 特征配置 ==========
features_config_path="configs/features_example.yaml"  # 特征配置文件路径（features.yaml）

# ========== 模型配置 ==========
model_name="mlp_classifier"                   # 模型名称（mlp_classifier 用于分类，mlp_regressor 用于回归）
model_hidden_dims="64 32"                     # 隐藏层维度（空格分隔，如："128 64 32"）
model_num_classes=3                           # 分类数量（分类任务，字典方式会自动推断）或输出数量（回归任务，通常为 1）
model_dropout=0.1                             # Dropout 比例
model_activation="relu"                       # 激活函数（relu, gelu, tanh）
model_batch_norm=false                        # 是否使用 BatchNorm（true/false）

# ========== 训练参数 ==========
num_epochs=10                                 # 训练轮数
batch_size=32                                 # 批次大小
learning_rate=0.001                           # 学习率
task_type="classification"                    # 任务类型（classification 分类, regression 回归）
optimizer="adam"                              # 优化器（adam, sgd）

# ========== 实验配置 ==========
experiment_name=""                            # 实验名称（留空则自动生成：baseline_YYYYMMDD_HHMMSS）
output_dir="outputs/baseline"                 # 输出目录

# ========== 分布式训练 ==========
use_ray=false                                 # 是否使用 Ray 分布式训练（true/false）

# ========== 环境配置 ==========
conda_env="bamboohepml"                       # Conda 环境名称
env_script=""                                 # 环境脚本路径（可选，如：/path/to/env.sh）

###### Part 3: 脚本执行（无需修改）######
set -e
set -x

# 环境设置
[ -n "$SLURM_JOB_ID" ] && srun -l hostname && /usr/bin/nvidia-smi -L && echo "GPU: ${CUDA_VISIBLE_DEVICES}"
[ -n "$env_script" ] && source "$env_script"
[ -n "$conda_env" ] && conda activate "$conda_env" 2>/dev/null || true

# 设置默认值
if [ -z "$experiment_name" ]; then
    experiment_name="baseline_$(date +%Y%m%d_%H%M%S)"
fi

# 创建目录
mkdir -p "${output_dir}" logs

# 将隐藏层维度字符串转换为 YAML 数组格式
hidden_dims_yaml="[$(echo $model_hidden_dims | sed 's/ /, /g')]"

# 生成临时 pipeline.yaml 配置文件
pipeline_config=$(mktemp /tmp/pipeline_XXXXXX.yaml)
cat > "$pipeline_config" <<EOF
# 自动生成的 Pipeline 配置
# 由 submit_baseline.sh 生成

# 数据配置
data:
  config_path: "${data_config_path}"
  source_path: "${data_train}"
  treename: "${data_treename}"

# 特征配置
features:
  config_path: "${features_config_path}"

# 模型配置
model:
  name: "${model_name}"
  params:
    hidden_dims: ${hidden_dims_yaml}
    num_classes: ${model_num_classes}
    dropout: ${model_dropout}
    activation: "${model_activation}"
    batch_norm: ${model_batch_norm}

# 训练配置
train:
  num_epochs: ${num_epochs}
  batch_size: ${batch_size}
  learning_rate: ${learning_rate}
  optimizer: "${optimizer}"
  task_type: "${task_type}"
EOF

echo "Generated pipeline config: $pipeline_config"
cat "$pipeline_config"

# 构建训练命令
CMD="bamboohepml train"
CMD="${CMD} -c ${pipeline_config}"
CMD="${CMD} --experiment-name ${experiment_name}"
CMD="${CMD} --output-dir ${output_dir}/${experiment_name}"
CMD="${CMD} --num-epochs ${num_epochs}"
CMD="${CMD} --batch-size ${batch_size}"
CMD="${CMD} --learning-rate ${learning_rate}"

# 如果启用 Ray，添加 --use-ray 标志
if [ "${use_ray}" = "true" ] || [ "${use_ray}" = "1" ]; then
    CMD="${CMD} --use-ray"
fi

# 传递额外参数
CMD="${CMD} $@"

# 执行训练
${CMD}

# 清理临时文件（可选，保留用于调试）
# rm -f "$pipeline_config"
# [ "$use_dict_mode" = true ] && rm -f "$temp_data_config"
