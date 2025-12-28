#!/usr/bin/env python
"""
联合测试：Data + Model

测试：
1. 数据加载和预处理
2. 模型创建和调用
3. 简单的训练循环（2 个 epoch）
4. 检查输入输出是否正确
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
os.environ["PYTHONPATH"] = str(project_root) + os.pathsep + os.environ.get("PYTHONPATH", "")

import tempfile  # noqa: E402

# 导入必须在路径设置之后
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from bamboohepml.data import DataConfig, DataSourceFactory, HEPDataset  # noqa: E402
from bamboohepml.models import get_model  # noqa: E402

# 测试文件路径
TEST_ROOT_FILE = (
    "/Users/physicsboy/Desktop/cepc_hss_scripts/Analysis/sample/tagging/input/TDR/train/Higgs/ss/merge_ss_0006.root"
)
TREE_NAME = "tree"


def print_section(title):
    """打印分节标题"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def create_test_config():
    """创建测试配置"""
    # 查找实际存在的分支
    source = DataSourceFactory.create(
        TEST_ROOT_FILE,
        treename=TREE_NAME,
        load_range=(0, 0.01),  # 只加载 1% 用于测试
    )
    branches = source.get_available_branches()

    def find_branch(pattern):
        matches = [b for b in branches if pattern.lower() in b.lower()]
        return matches[0] if matches else None

    # 查找分支
    jet_energy_branch = find_branch("jet_energy")
    jet_theta_branch = find_branch("jet_theta")
    jet_nparticles_branch = find_branch("jet_nparticles")

    # 创建配置
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
                "length": None,  # Event-level 特征，不需要 padding
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
        "observers": [
            jet_energy_branch or "jet_energy",
            jet_nparticles_branch or "jet_nparticles",
        ]
        if jet_energy_branch
        else [],
        "weights": None,
    }

    return config_dict


def test_classification():
    """测试分类任务"""
    print_section("测试 1: 分类任务（MLP Classifier）")

    if not os.path.exists(TEST_ROOT_FILE):
        print("⚠️  测试文件不存在，跳过此测试")
        return

    # 创建配置
    config_dict = create_test_config()

    # 创建临时配置文件
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_dict, f)
        config_file = f.name

    try:
        # 加载配置
        print("\n1. 加载数据配置...")
        data_config = DataConfig.load(config_file)
        print("   ✓ 配置加载成功")
        print(f"   输入变量: {list(data_config.input_dicts.keys())}")
        print(f"   标签: {data_config.label_names}")

        # 创建数据源
        print("\n2. 创建数据源...")
        source = DataSourceFactory.create(
            TEST_ROOT_FILE,
            treename=TREE_NAME,
            load_range=(0, 0.05),  # 只加载 5% 用于测试
        )
        print("   ✓ 数据源创建成功")

        # 创建数据集
        print("\n3. 创建数据集...")
        dataset = HEPDataset(
            data_source=source,
            data_config=data_config,
            for_training=True,
            shuffle=True,
        )
        print("   ✓ 数据集创建成功")

        # 创建 DataLoader
        print("\n4. 创建 DataLoader...")
        loader = DataLoader(dataset, batch_size=8, num_workers=0)
        print("   ✓ DataLoader 创建成功")

        # 获取一个样本以确定输入维度
        print("\n5. 检查数据格式...")
        sample = next(iter(dataset))
        print(f"   样本键: {list(sample.keys())}")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"     {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"     {key}: type={type(value).__name__}")

        # 确定输入维度（根据 DataConfig 的输入组名称）
        input_group_name = list(data_config.input_dicts.keys())[0]  # 第一个输入组
        input_key = "_" + input_group_name  # DataConfig 会在输入组名前加 '_'
        if input_key not in sample:
            # 尝试其他可能的键
            input_keys = [k for k in sample.keys() if k.startswith("_") and k != "_label_"]
            if input_keys:
                input_key = input_keys[0]
                print(f"   ⚠️  使用备用输入键: {input_key}")
            else:
                print(f"   ⚠️  未找到输入键，可用键: {list(sample.keys())}")
                return

        # 确定输入维度
        input_tensor = sample[input_key]
        if isinstance(input_tensor, torch.Tensor):
            if len(input_tensor.shape) == 1:
                input_dim = 1
            elif len(input_tensor.shape) == 2:
                input_dim = input_tensor.shape[1]
            else:
                # 如果是 3D (batch, features, length)，展平
                input_dim = (
                    input_tensor.shape[1] * input_tensor.shape[2] if len(input_tensor.shape) == 3 else input_tensor.shape[1]
                )
        else:
            input_dim = 1

        print(f"   输入键: {input_key}")
        print(f"   输入形状: {input_tensor.shape if isinstance(input_tensor, torch.Tensor) else 'N/A'}")
        print(f"   输入维度: {input_dim}")

        # 确定类别数（简化：使用 2 类分类）
        num_classes = 2
        print(f"   类别数: {num_classes}")

        # 创建模型
        print("\n6. 创建模型...")
        model = get_model(
            "mlp_classifier",
            task_type="classification",
            input_dim=input_dim,
            hidden_dims=[64, 32],
            num_classes=num_classes,
            dropout=0.1,
            activation="relu",
        )
        print("   ✓ 模型创建成功")
        print(f"   模型信息: {model.get_model_info()}")

        # 测试前向传播
        print("\n7. 测试前向传播...")
        batch = next(iter(loader))
        print(f"   Batch 键: {list(batch.keys())}")
        print(f"   输入形状: {batch[input_key].shape}")

        # 准备模型输入（展平如果必要）
        features = batch[input_key].float()
        if len(features.shape) == 3:
            # (batch, features, length) -> (batch, features * length)
            batch_size = features.shape[0]
            features = features.view(batch_size, -1)
        elif len(features.shape) == 2:
            # (batch, features) - 已经是正确的
            pass
        else:
            # (batch,) -> (batch, 1)
            features = features.unsqueeze(1)

        model_input = {"features": features}
        print(f"   模型输入形状: {model_input['features'].shape}")
        print(f"   实际输入维度: {model_input['features'].shape[1]}")

        # 更新模型输入维度（如果不同）
        if model_input["features"].shape[1] != input_dim:
            print("   ⚠️  输入维度不匹配，更新模型...")
            input_dim = model_input["features"].shape[1]
            model = get_model(
                "mlp_classifier",
                task_type="classification",
                input_dim=input_dim,
                hidden_dims=[64, 32],
                num_classes=num_classes,
                dropout=0.1,
                activation="relu",
            )
            print(f"   ✓ 模型已更新，输入维度: {input_dim}")

        # 前向传播
        model.eval()
        with torch.no_grad():
            output = model(model_input)
        print(f"   输出形状: {output.shape}")
        print(f"   输出值（前3个样本）: {output[:3]}")

        # 预测
        predictions = model.predict(model_input)
        print(f"   预测形状: {predictions.shape}")
        print(f"   预测值（前10个）: {predictions[:10]}")

        probabilities = model.predict_proba(model_input)
        print(f"   概率形状: {probabilities.shape}")
        print(f"   概率（前3个样本，所有类别）: {probabilities[:3]}")

        print("   ✓ 前向传播测试通过")

        # 简单的训练循环（2 个 epoch）
        print("\n8. 运行训练循环（2 个 epoch）...")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model.train()
        for epoch in range(2):
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(loader):
                # 准备输入（展平如果必要）
                features = batch[input_key].float()
                if len(features.shape) == 3:
                    batch_size = features.shape[0]
                    features = features.view(batch_size, -1)
                elif len(features.shape) == 1:
                    features = features.unsqueeze(1)
                model_input = {"features": features}

                # 准备标签（简化：将连续值转换为类别）
                labels = batch["_label_"].long()
                # 如果标签是连续值，转换为二分类（0 或 1）
                if labels.max() > 1:
                    labels = (labels > labels.median()).long()

                # 前向传播
                optimizer.zero_grad()
                output = model(model_input)

                # 计算损失
                loss = criterion(output, labels)

                # 反向传播
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                # 只处理前几个 batch 用于测试
                if batch_idx >= 5:
                    break

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            print(f"   Epoch {epoch + 1}/2: 平均损失 = {avg_loss:.4f}, Batch 数 = {num_batches}")

        print("   ✓ 训练循环完成")

        # 测试保存和加载
        print("\n9. 测试模型保存和加载...")
        save_dir = "test_checkpoints"
        os.makedirs(save_dir, exist_ok=True)

        model.save(save_dir, model_name="test_classifier")
        print("   ✓ 模型保存成功")

        from bamboohepml.models.common import MLPClassifier

        loaded_model = MLPClassifier.load(save_dir, model_name="test_classifier")
        print("   ✓ 模型加载成功")

        # 验证加载的模型
        with torch.no_grad():
            original_output = model(model_input)
            loaded_output = loaded_model(model_input)
            assert torch.allclose(original_output, loaded_output, atol=1e-5)
        print("   ✓ 加载的模型输出一致")

        # 清理
        import shutil

        shutil.rmtree(save_dir)

        print("\n✓ 分类任务测试完成")

    finally:
        if os.path.exists(config_file):
            os.unlink(config_file)


def test_regression():
    """测试回归任务"""
    print_section("测试 2: 回归任务（MLP Regressor）")

    if not os.path.exists(TEST_ROOT_FILE):
        print("⚠️  测试文件不存在，跳过此测试")
        return

    # 创建配置
    config_dict = create_test_config()

    # 创建临时配置文件
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_dict, f)
        config_file = f.name

    try:
        # 加载配置
        print("\n1. 加载数据配置...")
        data_config = DataConfig.load(config_file)
        print("   ✓ 配置加载成功")

        # 创建数据源和数据集
        print("\n2. 创建数据集...")
        source = DataSourceFactory.create(
            TEST_ROOT_FILE,
            treename=TREE_NAME,
            load_range=(0, 0.05),
        )
        dataset = HEPDataset(
            data_source=source,
            data_config=data_config,
            for_training=True,
            shuffle=True,
        )
        loader = DataLoader(dataset, batch_size=8, num_workers=0)
        print("   ✓ 数据集创建成功")

        # 获取输入维度
        sample = next(iter(dataset))
        input_key = "_features"
        if input_key not in sample:
            input_keys = [k for k in sample.keys() if k.startswith("_")]
            if input_keys:
                input_key = input_keys[0]
            else:
                print("   ⚠️  未找到输入键")
                return

        input_dim = sample[input_key].shape[-1] if len(sample[input_key].shape) > 1 else 1
        print(f"   输入维度: {input_dim}")

        # 创建回归模型
        print("\n3. 创建回归模型...")
        model = get_model(
            "mlp_regressor",
            task_type="regression",
            input_dim=input_dim,
            hidden_dims=[64, 32],
            num_outputs=1,
            dropout=0.1,
        )
        print("   ✓ 模型创建成功")

        # 测试前向传播
        print("\n4. 测试前向传播...")
        batch = next(iter(loader))

        # 准备输入（展平如果必要）
        features = batch[input_key].float()
        if len(features.shape) == 3:
            batch_size = features.shape[0]
            features = features.view(batch_size, -1)
        elif len(features.shape) == 1:
            features = features.unsqueeze(1)

        # 更新模型输入维度（如果不同）
        actual_input_dim = features.shape[1]
        if actual_input_dim != input_dim:
            print("   ⚠️  输入维度不匹配，更新模型...")
            input_dim = actual_input_dim
            model = get_model(
                "mlp_regressor",
                task_type="regression",
                input_dim=input_dim,
                hidden_dims=[64, 32],
                num_outputs=1,
                dropout=0.1,
            )

        model_input = {"features": features}

        model.eval()
        with torch.no_grad():
            output = model(model_input)
        print(f"   输入形状: {model_input['features'].shape}")
        print(f"   输出形状: {output.shape}")
        print(f"   输出值（前5个）: {output[:5].squeeze()}")

        predictions = model.predict(model_input)
        print(f"   预测形状: {predictions.shape}")
        print(f"   预测值（前5个）: {predictions[:5]}")

        print("   ✓ 前向传播测试通过")

        # 简单的训练循环（2 个 epoch）
        print("\n5. 运行训练循环（2 个 epoch）...")
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model.train()
        for epoch in range(2):
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(loader):
                # 准备输入（展平如果必要）
                features = batch[input_key].float()
                if len(features.shape) == 3:
                    batch_size = features.shape[0]
                    features = features.view(batch_size, -1)
                elif len(features.shape) == 1:
                    features = features.unsqueeze(1)
                model_input = {"features": features}

                # 使用标签作为目标（如果是连续值）
                labels = batch["_label_"].float()
                if len(labels.shape) > 1:
                    labels = labels.squeeze()

                optimizer.zero_grad()
                output = model(model_input)

                # 确保输出和标签形状匹配
                if output.dim() > 1:
                    output = output.squeeze()

                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                if batch_idx >= 5:
                    break

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            print(f"   Epoch {epoch + 1}/2: 平均损失 = {avg_loss:.4f}, Batch 数 = {num_batches}")

        print("   ✓ 训练循环完成")
        print("\n✓ 回归任务测试完成")

    finally:
        if os.path.exists(config_file):
            os.unlink(config_file)


def main():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("  联合测试：Data + Model")
    print("=" * 80)
    print(f"\n测试文件: {TEST_ROOT_FILE}")
    print(f"Tree 名称: {TREE_NAME}\n")

    if not os.path.exists(TEST_ROOT_FILE):
        print(f"❌ 错误: 测试文件不存在: {TEST_ROOT_FILE}")
        return

    try:
        # 测试 1: 分类任务
        test_classification()

        # 测试 2: 回归任务
        test_regression()

        print("\n" + "=" * 80)
        print("  ✓ 所有测试完成！")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
