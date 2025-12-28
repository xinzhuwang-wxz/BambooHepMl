"""
测试 Trainer Pipeline: Train -> Eval -> Test -> Infer

验证整个链路是否顺畅。
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
os.environ["PYTHONPATH"] = str(project_root) + os.pathsep + os.environ.get("PYTHONPATH", "")

# 导入必须在路径设置之后
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from bamboohepml.engine import EarlyStoppingCallback, Evaluator, LoggingCallback, Predictor, Trainer  # noqa: E402
from bamboohepml.models import get_model  # noqa: E402


def create_dummy_dataset(num_samples=1000, input_dim=10, num_classes=2):
    """创建虚拟数据集用于测试。"""
    # 生成随机数据
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))

    dataset = TensorDataset(X, y)
    return dataset


def test_train_eval_test_infer_pipeline():
    """测试完整的 Train -> Eval -> Test -> Infer 链路。"""
    print("=" * 80)
    print("测试 Trainer Pipeline: Train -> Eval -> Test -> Infer")
    print("=" * 80)

    # 1. 准备数据
    print("\n1. 准备数据...")
    train_dataset = create_dummy_dataset(num_samples=800, input_dim=10, num_classes=2)
    val_dataset = create_dummy_dataset(num_samples=100, input_dim=10, num_classes=2)
    test_dataset = create_dummy_dataset(num_samples=100, input_dim=10, num_classes=2)

    # 创建 DataLoader（需要适配 HEPDataset 格式）
    def collate_fn(batch):
        """将 TensorDataset 转换为 HEPDataset 格式。"""
        X, y = zip(*batch)
        X = torch.stack(X)
        y = torch.stack(y)
        return {
            "_features": X,
            "_label_": y,
        }

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    print(f"   ✓ 训练集: {len(train_dataset)} 样本")
    print(f"   ✓ 验证集: {len(val_dataset)} 样本")
    print(f"   ✓ 测试集: {len(test_dataset)} 样本")

    # 2. 创建模型
    print("\n2. 创建模型...")
    model = get_model(
        "mlp_classifier",
        task_type="classification",
        input_dim=10,
        hidden_dims=[32, 16],
        num_classes=2,
        dropout=0.1,
    )
    print("   ✓ 模型创建成功")

    # 3. 创建 Trainer
    print("\n3. 创建 Trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        task_type="supervised",
        callbacks=[
            LoggingCallback(),
            EarlyStoppingCallback(monitor="val_loss", patience=5, min_delta=0.001),
        ],
    )
    print("   ✓ Trainer 创建成功")

    # 4. 训练
    print("\n4. 训练模型...")
    history = trainer.fit(
        num_epochs=10,
        save_dir="./test_checkpoints",
        save_best=True,
        monitor="val_loss",
    )
    print("   ✓ 训练完成")
    print(f"   最佳 epoch: {history['best_epoch']}")
    print(f"   最佳值: {history['best_value']:.4f}")

    # 5. 验证（使用 Trainer.validate）
    print("\n5. 验证模型（Trainer.validate）...")
    val_metrics = trainer.validate()
    print("   ✓ 验证完成")
    print(f"   验证指标: {val_metrics}")

    # 6. 测试（使用 Trainer.test）
    print("\n6. 测试模型（Trainer.test）...")
    test_metrics = trainer.test()
    print("   ✓ 测试完成")
    print(f"   测试指标: {test_metrics}")

    # 7. 使用 Evaluator 评估
    print("\n7. 使用 Evaluator 评估...")
    evaluator = Evaluator(task_type="classification")
    eval_metrics = evaluator.evaluate(
        model=model,
        dataloader=test_loader,
        loss_fn=nn.CrossEntropyLoss(),
    )
    print("   ✓ 评估完成")
    print(f"   评估指标: {eval_metrics}")

    # 8. 推理（使用 Predictor）
    print("\n8. 推理预测（Predictor.predict）...")
    predictor = Predictor(model)
    predictions = predictor.predict(
        dataloader=test_loader,
        return_probabilities=True,
    )
    print("   ✓ 推理完成")
    print(f"   预测样本数: {len(predictions)}")
    print(f"   前 3 个预测: {predictions[:3]}")

    # 9. 检查链路
    print("\n9. 检查链路...")
    checks = []

    # Check 1: Train -> Eval
    checks.append(("Train -> Eval", "val_loss" in history["history"]))

    # Check 2: Eval -> Test
    checks.append(("Eval -> Test", "loss" in test_metrics))

    # Check 3: Test -> Infer
    checks.append(("Test -> Infer", len(predictions) == len(test_dataset)))

    # Check 4: Predictor 格式
    checks.append(("Predictor 格式", "prediction" in predictions[0]))

    for check_name, check_result in checks:
        status = "✓" if check_result else "✗"
        print(f"   {status} {check_name}: {'通过' if check_result else '失败'}")

    # 10. 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    all_passed = all(result for _, result in checks)
    if all_passed:
        print("✓ 所有链路测试通过！")
        print("\n链路流程:")
        print("  1. Train: Trainer.fit() -> 训练模型")
        print("  2. Eval: Trainer.validate() -> 验证模型")
        print("  3. Test: Trainer.test() -> 测试模型")
        print("  4. Infer: Predictor.predict() -> 推理预测")
    else:
        print("✗ 部分链路测试失败")

    return all_passed


if __name__ == "__main__":
    try:
        success = test_train_eval_test_infer_pipeline()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
