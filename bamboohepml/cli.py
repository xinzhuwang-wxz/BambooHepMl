"""
BambooHepMl CLI

使用 Typer 实现命令行接口，参考 Made-With-ML。
"""

from pathlib import Path
from typing import Annotated, Optional

import typer

from .config import logger
from .scheduler import LocalScheduler, SLURMScheduler

# Tasks are called through schedulers, not directly imported

# 初始化 Typer CLI app
app = typer.Typer(help="BambooHepMl: 高能物理机器学习框架")

# 添加子命令组
serve_app = typer.Typer(help="服务命令")
app.add_typer(serve_app, name="serve")


@app.command()
def train(
    pipeline_config: Annotated[str, typer.Option("-c", "--config", help="Pipeline 配置文件路径")],
    experiment_name: Annotated[Optional[str], typer.Option("--experiment-name", help="实验名称（用于 MLflow）")] = None,
    num_epochs: Annotated[Optional[int], typer.Option("--num-epochs", help="训练轮数（覆盖配置）")] = None,
    batch_size: Annotated[Optional[int], typer.Option("--batch-size", help="批次大小（覆盖配置）")] = None,
    learning_rate: Annotated[Optional[float], typer.Option("--learning-rate", help="学习率（覆盖配置）")] = None,
    output_dir: Annotated[Optional[str], typer.Option("-o", "--output-dir", help="输出目录")] = None,
    use_ray: Annotated[bool, typer.Option("--use-ray/--no-ray", help="是否使用 Ray 分布式训练")] = False,
    num_workers: Annotated[int, typer.Option("--num-workers", help="Ray worker 数量")] = 1,
    gpu_per_worker: Annotated[int, typer.Option("--gpu-per-worker", help="每个 worker 的 GPU 数量")] = 0,
    scheduler: Annotated[str, typer.Option("--scheduler", help="调度器类型：local 或 slurm")] = "local",
    slurm_config: Annotated[Optional[str], typer.Option("--slurm-config", help="SLURM 配置文件路径（仅 slurm 调度器）")] = None,
) -> None:
    """训练模型。

    示例:
        bamboohepml train -c configs/pipeline.yaml --experiment-name my_exp
        bamboohepml train -c configs/pipeline.yaml --scheduler slurm --slurm-config slurm_config.sh
    """
    logger.info("=" * 80)
    logger.info("BambooHepMl Train Command")
    logger.info("=" * 80)

    # 验证配置文件存在
    if not Path(pipeline_config).exists():
        logger.error(f"Pipeline config file not found: {pipeline_config}")
        raise typer.Exit(1)

    # 选择调度器
    if scheduler == "local":
        scheduler_instance = LocalScheduler()
    elif scheduler == "slurm":
        scheduler_instance = SLURMScheduler(slurm_config_path=slurm_config)
    else:
        logger.error(f"Unknown scheduler: {scheduler}. Must be 'local' or 'slurm'")
        raise typer.Exit(1)

    # 准备训练参数
    train_kwargs = {
        "pipeline_config_path": pipeline_config,
        "experiment_name": experiment_name,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "output_dir": output_dir,
        "use_ray": use_ray,
        "num_workers": num_workers,
        "gpu_per_worker": gpu_per_worker,
    }

    # 使用调度器提交任务
    scheduler_instance.submit_train(**train_kwargs)


@app.command()
def predict(
    pipeline_config: Annotated[str, typer.Option("-c", "--config", help="Pipeline 配置文件路径")],
    model_path: Annotated[str, typer.Option("-m", "--model", help="模型文件路径")],
    output_path: Annotated[Optional[str], typer.Option("-o", "--output", help="输出文件路径")] = None,
    batch_size: Annotated[int, typer.Option("--batch-size", help="批次大小")] = 32,
    return_probabilities: Annotated[bool, typer.Option("--probabilities/--no-probabilities", help="是否返回概率")] = False,
    scheduler: Annotated[str, typer.Option("--scheduler", help="调度器类型：local 或 slurm")] = "local",
    slurm_config: Annotated[Optional[str], typer.Option("--slurm-config", help="SLURM 配置文件路径（仅 slurm 调度器）")] = None,
) -> None:
    """使用模型进行预测。

    示例:
        bamboohepml predict -c configs/pipeline.yaml -m outputs/model.pt -o predictions.json
    """
    logger.info("=" * 80)
    logger.info("BambooHepMl Predict Command")
    logger.info("=" * 80)

    # 验证文件存在
    if not Path(pipeline_config).exists():
        logger.error(f"Pipeline config file not found: {pipeline_config}")
        raise typer.Exit(1)

    if not Path(model_path).exists():
        logger.error(f"Model file not found: {model_path}")
        raise typer.Exit(1)

    # 选择调度器
    if scheduler == "local":
        scheduler_instance = LocalScheduler()
    elif scheduler == "slurm":
        scheduler_instance = SLURMScheduler(slurm_config_path=slurm_config)
    else:
        logger.error(f"Unknown scheduler: {scheduler}. Must be 'local' or 'slurm'")
        raise typer.Exit(1)

    # 准备预测参数
    predict_kwargs = {
        "pipeline_config_path": pipeline_config,
        "model_path": model_path,
        "output_path": output_path,
        "batch_size": batch_size,
        "return_probabilities": return_probabilities,
    }

    # 使用调度器提交任务
    scheduler_instance.submit_predict(**predict_kwargs)


@app.command()
def export(
    pipeline_config: Annotated[str, typer.Option("-c", "--config", help="Pipeline 配置文件路径")],
    model_path: Annotated[str, typer.Option("-m", "--model", help="模型文件路径")],
    output_path: Annotated[str, typer.Option("-o", "--output", help="输出 ONNX 文件路径")],
    input_shape: Annotated[Optional[str], typer.Option("--input-shape", help="输入形状，格式：'batch,features' 或 'features'")] = None,
    opset_version: Annotated[int, typer.Option("--opset-version", help="ONNX opset 版本")] = 11,
    scheduler: Annotated[str, typer.Option("--scheduler", help="调度器类型：local 或 slurm")] = "local",
    slurm_config: Annotated[Optional[str], typer.Option("--slurm-config", help="SLURM 配置文件路径（仅 slurm 调度器）")] = None,
) -> None:
    """导出模型为 ONNX 格式。

    示例:
        bamboohepml export -c configs/pipeline.yaml -m outputs/model.pt -o model.onnx
    """
    logger.info("=" * 80)
    logger.info("BambooHepMl Export Command")
    logger.info("=" * 80)

    # 验证文件存在
    if not Path(pipeline_config).exists():
        logger.error(f"Pipeline config file not found: {pipeline_config}")
        raise typer.Exit(1)

    if not Path(model_path).exists():
        logger.error(f"Model file not found: {model_path}")
        raise typer.Exit(1)

    # 解析 input_shape
    parsed_input_shape = None
    if input_shape:
        try:
            parsed_input_shape = tuple(map(int, input_shape.split(",")))
        except ValueError:
            logger.error(f"Invalid input_shape format: {input_shape}. Expected format: 'batch,features' or 'features'")
            raise typer.Exit(1)

    # 选择调度器
    if scheduler == "local":
        scheduler_instance = LocalScheduler()
    elif scheduler == "slurm":
        scheduler_instance = SLURMScheduler(slurm_config_path=slurm_config)
    else:
        logger.error(f"Unknown scheduler: {scheduler}. Must be 'local' or 'slurm'")
        raise typer.Exit(1)

    # 准备导出参数
    export_kwargs = {
        "pipeline_config_path": pipeline_config,
        "model_path": model_path,
        "output_path": output_path,
        "input_shape": parsed_input_shape,
        "opset_version": opset_version,
    }

    # 使用调度器提交任务
    scheduler_instance.submit_export(**export_kwargs)


@app.command()
def inspect(
    pipeline_config: Annotated[str, typer.Option("-c", "--config", help="Pipeline 配置文件路径")],
    output_path: Annotated[Optional[str], typer.Option("-o", "--output", help="输出文件路径")] = None,
    num_samples: Annotated[int, typer.Option("--num-samples", help="检查的样本数量")] = 1000,
    inspect_data: Annotated[bool, typer.Option("--inspect-data/--no-inspect-data", help="是否检查数据")] = True,
    inspect_features: Annotated[bool, typer.Option("--inspect-features/--no-inspect-features", help="是否检查特征")] = True,
    scheduler: Annotated[str, typer.Option("--scheduler", help="调度器类型：local 或 slurm")] = "local",
    slurm_config: Annotated[Optional[str], typer.Option("--slurm-config", help="SLURM 配置文件路径（仅 slurm 调度器）")] = None,
) -> None:
    """检查数据和特征。

    示例:
        bamboohepml inspect -c configs/pipeline.yaml -o inspection.json
    """
    logger.info("=" * 80)
    logger.info("BambooHepMl Inspect Command")
    logger.info("=" * 80)

    # 验证配置文件存在
    if not Path(pipeline_config).exists():
        logger.error(f"Pipeline config file not found: {pipeline_config}")
        raise typer.Exit(1)

    # 选择调度器
    if scheduler == "local":
        scheduler_instance = LocalScheduler()
    elif scheduler == "slurm":
        scheduler_instance = SLURMScheduler(slurm_config_path=slurm_config)
    else:
        logger.error(f"Unknown scheduler: {scheduler}. Must be 'local' or 'slurm'")
        raise typer.Exit(1)

    # 准备检查参数
    inspect_kwargs = {
        "pipeline_config_path": pipeline_config,
        "output_path": output_path,
        "num_samples": num_samples,
        "inspect_data": inspect_data,
        "inspect_features": inspect_features,
    }

    # 使用调度器提交任务
    scheduler_instance.submit_inspect(**inspect_kwargs)


# 导入服务命令（必须在 app 定义之后）
from .serve.cli import fastapi as serve_fastapi_cmd  # noqa: E402
from .serve.cli import ray as serve_ray_cmd  # noqa: E402

serve_app.command(name="fastapi")(serve_fastapi_cmd)
serve_app.command(name="ray")(serve_ray_cmd)


def main():
    """CLI 入口点。"""
    app()


if __name__ == "__main__":
    main()
