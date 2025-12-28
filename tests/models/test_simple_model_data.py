#!/usr/bin/env python
"""
简化版联合测试：Data + Model

简单测试模型的调用，运行 2 个 epoch
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
os.environ["PYTHONPATH"] = str(project_root) + os.pathsep + os.environ.get("PYTHONPATH", "")

import tempfile  # noqa: E402

import numpy as np  # noqa: E402

# 导入必须在路径设置之后
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from bamboohepml.data import DataConfig, DataSourceFactory, HEPDataset  # noqa: E402
from bamboohepml.models import get_model  # noqa: E402

# 测试文件路径
TEST_ROOT_FILE = "/Users/physicsboy/Desktop/cepc_hss_scripts/Analysis/sample/tagging/input/TDR/train/Higgs/ss/merge_ss_0006.root"
TREE_NAME = "tree"


def main():
    print("=" * 80)
    print("简化版联合测试：Data + Model")
    print("=" * 80)

    if not os.path.exists(TEST_ROOT_FILE):
        print(f"❌ 测试文件不存在: {TEST_ROOT_FILE}")
        return

    # 1. 创建数据配置
    print("\n1. 创建数据配置...")
    source = DataSourceFactory.create(
        TEST_ROOT_FILE,
        treename=TREE_NAME,
        load_range=(0, 0.01),
    )
    branches = source.get_available_branches()

    def find_branch(pattern):
        matches = [b for b in branches if pattern.lower() in b.lower()]
        return matches[0] if matches else None

    jet_energy_branch = find_branch("jet_energy")
    jet_theta_branch = find_branch("jet_theta")
    jet_nparticles_branch = find_branch("jet_nparticles")

    config_dict = {
        "treename": TREE_NAME,
        "selection": None,
        "new_variables": {},
        "preprocess": {
            "method": "manual",
            "data_fraction": 0.5,
        },
        "inputs": {
            "features": {
                "length": None,  # Event-level 特征
                "pad_mode": "constant",
                "vars": [
                    [jet_energy_branch or "jet_energy", None, 1, 0, 1000, 0],
                    [jet_nparticles_branch or "jet_nparticles", None, 1, 0, 100, 0],
                ],
            },
        },
        "labels": {
            "type": "simple",
            "value": [jet_theta_branch or "jet_theta"] if jet_theta_branch else ["jet_energy"],
        },
        "observers": [],
        "weights": None,
    }

    # 创建临时配置文件
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_dict, f)
        config_file = f.name

    try:
        # 2. 加载配置
        print("2. 加载数据配置...")
        data_config = DataConfig.load(config_file)
        print("   ✓ 配置加载成功")
        print(f"   输入组: {list(data_config.input_dicts.keys())}")

        # 3. 创建数据集
        print("\n3. 创建数据集...")
        source = DataSourceFactory.create(
            TEST_ROOT_FILE,
            treename=TREE_NAME,
            load_range=(0, 0.05),  # 5% 数据
        )
        dataset = HEPDataset(
            data_source=source,
            data_config=data_config,
            for_training=True,
            shuffle=True,
        )
        loader = DataLoader(dataset, batch_size=8, num_workers=0)
        print("   ✓ 数据集创建成功")

        # 4. 检查数据格式
        print("\n4. 检查数据格式...")
        sample = next(iter(dataset))
        print(f"   样本键: {list(sample.keys())}")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"     {key}: torch.Tensor, shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, np.ndarray):
                print(f"     {key}: np.ndarray, shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"     {key}: {type(value).__name__}")

        input_group_name = list(data_config.input_dicts.keys())[0]
        input_key = "_" + input_group_name
        if input_key not in sample:
            input_keys = [k for k in sample.keys() if k.startswith("_") and k != "_label_"]
            if input_keys:
                input_key = input_keys[0]
                print(f"   ⚠️  使用备用输入键: {input_key}")
            else:
                print("   ❌ 未找到输入键")
                return

        print(f"   输入键: {input_key}")
        input_value = sample[input_key]

        # 转换为 torch.Tensor（如果需要）
        if isinstance(input_value, np.ndarray):
            input_tensor = torch.from_numpy(input_value)
        elif isinstance(input_value, torch.Tensor):
            input_tensor = input_value
        else:
            input_tensor = torch.tensor(input_value)

        print(f"   单个样本输入形状: {input_tensor.shape}")

        # 确定输入维度（最后一个维度是特征维度）
        # 注意：这里处理的是单个样本，不是 batch
        if len(input_tensor.shape) == 0:
            # 标量 -> 1
            input_dim = 1
        elif len(input_tensor.shape) == 1:
            # 1D: (features,) -> features
            input_dim = input_tensor.shape[0]
        elif len(input_tensor.shape) == 2:
            # 2D: (features, length) -> features * length（对于单个样本）
            input_dim = input_tensor.shape[0] * input_tensor.shape[1]
        else:
            # 其他情况：展平所有维度（除了第一个 batch 维度）
            input_dim = int(torch.prod(torch.tensor(input_tensor.shape)))

        print(f"   计算的输入维度: {input_dim}")

        # 5. 创建模型
        print("\n5. 创建模型...")
        model = get_model(
            "mlp_classifier",
            task_type="classification",
            input_dim=input_dim,
            hidden_dims=[64, 32],
            num_classes=2,
            dropout=0.1,
        )
        print("   ✓ 模型创建成功")
        info = model.get_model_info()
        print(f"   总参数: {info['total_parameters']}")
        print(f"   可训练参数: {info['trainable_parameters']}")

        # 6. 测试前向传播
        print("\n6. 测试前向传播...")
        batch = next(iter(loader))
        print(f"   Batch 键: {list(batch.keys())}")

        # 准备输入（确保是 torch.Tensor）
        features = batch[input_key]
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        features = features.float()

        print(f"   原始输入形状: {features.shape}")

        # 处理不同形状的输入
        if len(features.shape) == 3:
            # 3D: (batch, features, length) -> 展平为 (batch, features * length)
            batch_size = features.shape[0]
            features = features.view(batch_size, -1)
            # 更新模型
            if features.shape[1] != input_dim:
                print("   ⚠️  输入维度不匹配，更新模型...")
                input_dim = features.shape[1]
                model = get_model(
                    "mlp_classifier",
                    task_type="classification",
                    input_dim=input_dim,
                    hidden_dims=[64, 32],
                    num_classes=2,
                    dropout=0.1,
                )
                print(f"   ✓ 模型已更新，输入维度: {input_dim}")
        elif len(features.shape) == 2:
            # 2D: (batch, features) -> 直接使用
            # 检查维度是否匹配
            if features.shape[1] != input_dim:
                print(f"   ⚠️  输入维度不匹配: batch={features.shape[1]}, model={input_dim}")
                print("   更新模型以匹配 batch 输入维度...")
                input_dim = features.shape[1]
                model = get_model(
                    "mlp_classifier",
                    task_type="classification",
                    input_dim=input_dim,
                    hidden_dims=[64, 32],
                    num_classes=2,
                    dropout=0.1,
                )
                print(f"   ✓ 模型已更新，输入维度: {input_dim}")
        elif len(features.shape) == 1:
            # 1D: (features,) -> 添加 batch 维度 -> (1, features)
            features = features.unsqueeze(0)
            if features.shape[1] != input_dim:
                print("   ⚠️  输入维度不匹配，更新模型...")
                input_dim = features.shape[1]
                model = get_model(
                    "mlp_classifier",
                    task_type="classification",
                    input_dim=input_dim,
                    hidden_dims=[64, 32],
                    num_classes=2,
                    dropout=0.1,
                )
                print(f"   ✓ 模型已更新，输入维度: {input_dim}")

        model_input = {"features": features}
        print(f"   模型输入形状: {model_input['features'].shape}")
        print(f"   模型输入类型: {type(model_input['features'])}")

        model.eval()
        with torch.no_grad():
            output = model(model_input)
        print(f"   输出形状: {output.shape}")
        print(f"   输出值（前3个）: {output[:3]}")

        predictions = model.predict(model_input)
        print(f"   预测形状: {predictions.shape}")
        print(f"   预测值（前5个）: {predictions[:5]}")

        print("   ✓ 前向传播测试通过")

        # 7. 训练循环（2 个 epoch）
        print("\n7. 运行训练循环（2 个 epoch）...")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model.train()
        for epoch in range(2):
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(loader):
                # 准备输入（确保是 torch.Tensor）
                features = batch[input_key]
                if isinstance(features, np.ndarray):
                    features = torch.from_numpy(features)
                features = features.float()

                if len(features.shape) == 3:
                    batch_size = features.shape[0]
                    features = features.view(batch_size, -1)
                elif len(features.shape) == 1:
                    features = features.unsqueeze(1)
                model_input = {"features": features}

                # 准备标签（转换为二分类）
                labels = batch["_label_"]
                if isinstance(labels, np.ndarray):
                    labels = torch.from_numpy(labels)
                labels = labels.long()
                if labels.max() > 1:
                    labels = (labels > labels.median()).long()

                # 前向传播
                optimizer.zero_grad()
                output = model(model_input)
                loss = criterion(output, labels)

                # 反向传播
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                # 打印第一个 batch 的详细信息
                if batch_idx == 0:
                    print(f"   Batch {batch_idx + 1}:")
                    print(f"     输入形状: {model_input['features'].shape}")
                    print(f"     标签形状: {labels.shape}")
                    print(f"     输出形状: {output.shape}")
                    print(f"     损失: {loss.item():.4f}")

                # 只处理前 5 个 batch
                if batch_idx >= 4:
                    break

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            print(f"   Epoch {epoch + 1}/2: 平均损失 = {avg_loss:.4f}, Batch 数 = {num_batches}")

        print("   ✓ 训练循环完成")

        # 8. 测试预测
        print("\n8. 测试预测...")
        model.eval()
        with torch.no_grad():
            batch = next(iter(loader))
            features = batch[input_key]
            if isinstance(features, np.ndarray):
                features = torch.from_numpy(features)
            features = features.float()

            if len(features.shape) == 3:
                batch_size = features.shape[0]
                features = features.view(batch_size, -1)
            elif len(features.shape) == 1:
                features = features.unsqueeze(1)
            model_input = {"features": features}

            predictions = model.predict(model_input)
            probabilities = model.predict_proba(model_input)

            print(f"   预测形状: {predictions.shape}")
            print(f"   概率形状: {probabilities.shape}")
            print(f"   预测值（前10个）: {predictions[:10]}")
            print(f"   概率（前3个样本）: {probabilities[:3]}")

        print("   ✓ 预测测试通过")

        print("\n" + "=" * 80)
        print("  ✓ 所有测试完成！")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        if os.path.exists(config_file):
            os.unlink(config_file)


if __name__ == "__main__":
    main()
