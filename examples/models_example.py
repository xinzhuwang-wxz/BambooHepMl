"""
模型使用示例

展示如何使用 BambooHepMl 的模型层：
1. 分类任务示例
2. 回归任务示例
3. Finetune 示例
"""
import torch
from bamboohepml.models import get_model, BaseModel
from bamboohepml.models.common import MLPClassifier, MLPRegressor


def example_classification():
    """示例 1: 分类任务"""
    print("=" * 80)
    print("示例 1: 分类任务")
    print("=" * 80)
    
    # 方法 1: 使用 get_model 工厂函数
    model = get_model(
        'mlp_classifier',
        task_type='classification',
        input_dim=128,
        hidden_dims=[256, 128, 64],
        num_classes=10,
        dropout=0.1,
        activation='relu',
        batch_norm=True
    )
    
    print(f"\n模型类型: {model.__class__.__name__}")
    print(f"模型信息: {model.get_model_info()}")
    
    # 创建示例数据
    batch = {
        'features': torch.randn(32, 128)  # (batch_size, input_dim)
    }
    
    # 前向传播
    logits = model(batch)
    print(f"\n输入形状: {batch['features'].shape}")
    print(f"输出形状 (logits): {logits.shape}")
    
    # 预测
    predictions = model.predict(batch)
    print(f"预测形状: {predictions.shape}")
    print(f"预测值（前5个）: {predictions[:5]}")
    
    # 预测概率
    probabilities = model.predict_proba(batch)
    print(f"概率形状: {probabilities.shape}")
    print(f"概率（前5个样本，前3个类别）: {probabilities[:5, :3]}")
    
    # 保存模型
    print("\n保存模型...")
    model.save('checkpoints', model_name='classifier_example')
    
    # 加载模型
    print("加载模型...")
    loaded_model = MLPClassifier.load('checkpoints', model_name='classifier_example')
    print(f"加载成功: {loaded_model.__class__.__name__}")
    
    print("\n✓ 分类任务示例完成\n")


def example_regression():
    """示例 2: 回归任务"""
    print("=" * 80)
    print("示例 2: 回归任务")
    print("=" * 80)
    
    # 方法 1: 使用 get_model 工厂函数
    model = get_model(
        'mlp_regressor',
        task_type='regression',
        input_dim=128,
        hidden_dims=[256, 128, 64],
        num_outputs=1,
        dropout=0.1,
        activation='relu',
        batch_norm=True
    )
    
    print(f"\n模型类型: {model.__class__.__name__}")
    print(f"模型信息: {model.get_model_info()}")
    
    # 创建示例数据
    batch = {
        'features': torch.randn(32, 128)  # (batch_size, input_dim)
    }
    
    # 前向传播
    predictions = model(batch)
    print(f"\n输入形状: {batch['features'].shape}")
    print(f"输出形状: {predictions.shape}")
    
    # 预测
    predictions = model.predict(batch)
    print(f"预测形状: {predictions.shape}")
    print(f"预测值（前5个）: {predictions[:5]}")
    
    # 回归任务不支持 predict_proba
    try:
        model.predict_proba(batch)
    except NotImplementedError as e:
        print(f"\n回归任务不支持 predict_proba: {e}")
    
    # 保存模型
    print("\n保存模型...")
    model.save('checkpoints', model_name='regressor_example')
    
    # 加载模型
    print("加载模型...")
    loaded_model = MLPRegressor.load('checkpoints', model_name='regressor_example')
    print(f"加载成功: {loaded_model.__class__.__name__}")
    
    print("\n✓ 回归任务示例完成\n")


def example_finetune():
    """示例 3: Finetune（冻结/解冻层）"""
    print("=" * 80)
    print("示例 3: Finetune（冻结/解冻层）")
    print("=" * 80)
    
    # 创建模型
    model = get_model(
        'mlp_classifier',
        task_type='classification',
        input_dim=128,
        hidden_dims=[256, 128, 64],
        num_classes=10,
        dropout=0.1
    )
    
    print("\n初始状态:")
    info = model.get_model_info()
    print(f"  总参数: {info['total_parameters']}")
    print(f"  可训练参数: {info['trainable_parameters']}")
    print(f"  冻结层: {info['frozen_layers']}")
    
    # 冻结所有层（除了分类头）
    print("\n冻结所有层（除了分类头）...")
    model.freeze_layers(freeze_all=True)
    
    info = model.get_model_info()
    print(f"  可训练参数: {info['trainable_parameters']}")
    print(f"  冻结层数量: {len(info['frozen_layers'])}")
    print(f"  冻结层（前5个）: {info['frozen_layers'][:5]}")
    
    # 解冻最后两层
    print("\n解冻最后两层...")
    model.unfreeze_layers(layer_names=['network.4', 'network.5'])
    
    info = model.get_model_info()
    print(f"  可训练参数: {info['trainable_parameters']}")
    print(f"  冻结层数量: {len(info['frozen_layers'])}")
    
    # 解冻所有层
    print("\n解冻所有层...")
    model.unfreeze_layers(unfreeze_all=True)
    
    info = model.get_model_info()
    print(f"  可训练参数: {info['trainable_parameters']}")
    print(f"  冻结层: {info['frozen_layers']}")
    
    print("\n✓ Finetune 示例完成\n")


def example_direct_instantiation():
    """示例 4: 直接实例化（不使用 get_model）"""
    print("=" * 80)
    print("示例 4: 直接实例化模型")
    print("=" * 80)
    
    # 直接实例化分类模型
    classifier = MLPClassifier(
        input_dim=128,
        hidden_dims=[256, 128],
        num_classes=5,
        dropout=0.2
    )
    
    print(f"模型类型: {classifier.__class__.__name__}")
    print(f"类别数: {classifier.num_classes}")
    
    # 直接实例化回归模型
    regressor = MLPRegressor(
        input_dim=128,
        hidden_dims=[256, 128],
        num_outputs=3,  # 多输出回归
        dropout=0.2
    )
    
    print(f"模型类型: {regressor.__class__.__name__}")
    print(f"输出数: {regressor.num_outputs}")
    
    print("\n✓ 直接实例化示例完成\n")


if __name__ == "__main__":
    import os
    import shutil
    
    # 清理旧的 checkpoints
    if os.path.exists('checkpoints'):
        shutil.rmtree('checkpoints')
    
    # 运行示例
    example_classification()
    example_regression()
    example_finetune()
    example_direct_instantiation()
    
    print("=" * 80)
    print("所有示例完成！")
    print("=" * 80)

