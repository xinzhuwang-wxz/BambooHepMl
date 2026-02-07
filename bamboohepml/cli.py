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
    task: Annotated[
        Optional[str],
        typer.Option("--task", help="任务类型: classification 或 regression"),
    ] = None,
    model: Annotated[Optional[str], typer.Option("--model", help="模型类型: torch 或 xgboost")] = None,
    all_experiments: Annotated[bool, typer.Option("--all/--no-all", help="运行所有 task×model 组合")] = False,
    runs: Annotated[int, typer.Option("--runs", help="每个实验组合重复运行次数")] = 1,
    experiment_name: Annotated[Optional[str], typer.Option("--experiment-name", help="实验名称（用于 MLflow）")] = None,
    num_epochs: Annotated[Optional[int], typer.Option("--num-epochs", help="训练轮数（覆盖配置）")] = None,
    batch_size: Annotated[Optional[int], typer.Option("--batch-size", help="批次大小（覆盖配置）")] = None,
    learning_rate: Annotated[Optional[float], typer.Option("--learning-rate", help="学习率（覆盖配置）")] = None,
    output_dir: Annotated[Optional[str], typer.Option("-o", "--output-dir", help="输出目录")] = None,
    use_ray: Annotated[bool, typer.Option("--use-ray/--no-ray", help="是否使用 Ray 分布式训练")] = False,
    num_workers: Annotated[int, typer.Option("--num-workers", help="Ray worker 数量")] = 1,
    gpu_per_worker: Annotated[int, typer.Option("--gpu-per-worker", help="每个 worker 的 GPU 数量")] = 0,
    scheduler: Annotated[str, typer.Option("--scheduler", help="调度器类型：local 或 slurm")] = "local",
    slurm_config: Annotated[
        Optional[str],
        typer.Option("--slurm-config", help="SLURM 配置文件路径（仅 slurm 调度器）"),
    ] = None,
) -> None:
    """训练模型。

    支持多实验批量运行：
        bamboohepml train -c configs/pipeline_edm4hep.yaml --all --runs 2
        bamboohepml train -c configs/pipeline_edm4hep.yaml --task classification --model torch
        bamboohepml train -c configs/pipeline_edm4hep.yaml --experiment-name my_exp
    """
    logger.info("=" * 80)
    logger.info("BambooHepMl Train Command")
    logger.info("=" * 80)

    # 验证配置文件存在
    if not Path(pipeline_config).exists():
        logger.error(f"Pipeline config file not found: {pipeline_config}")
        raise typer.Exit(1)

    # 验证参数组合
    if not all_experiments and not task and not model:
        # 无 --task/--model/--all：单次运行（旧行为，兼容）
        pass
    elif all_experiments:
        if task or model:
            logger.error("--all cannot be used with --task or --model")
            raise typer.Exit(1)
    else:
        # 必须同时指定 --task 和 --model
        if not task or not model:
            logger.error("Must specify both --task and --model, or use --all")
            raise typer.Exit(1)
        if task not in ("classification", "regression"):
            logger.error(f"Invalid --task: {task}. Must be 'classification' or 'regression'")
            raise typer.Exit(1)
        if model not in ("torch", "xgboost"):
            logger.error(f"Invalid --model: {model}. Must be 'torch' or 'xgboost'")
            raise typer.Exit(1)

    # 选择调度器
    if scheduler == "local":
        scheduler_instance = LocalScheduler()
    elif scheduler == "slurm":
        scheduler_instance = SLURMScheduler(slurm_config_path=slurm_config)
    else:
        logger.error(f"Unknown scheduler: {scheduler}. Must be 'local' or 'slurm'")
        raise typer.Exit(1)

    # 构建实验列表
    if all_experiments:
        experiments = [
            ("classification", "torch"),
            ("classification", "xgboost"),
            ("regression", "torch"),
            ("regression", "xgboost"),
        ]
    elif task and model:
        experiments = [(task, model)]
    else:
        # 单次运行（旧行为）— 不指定 task/model
        experiments = [(None, None)]

    # 运行实验
    for task_type, model_type in experiments:
        for run_idx in range(1, runs + 1):
            run_label = f"{task_type or 'default'}_{model_type or 'default'}_run{run_idx}"
            logger.info(f"Starting experiment: {run_label}")

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
                "task_type": task_type,
                "model_type": model_type,
                "run_index": run_idx,
            }

            # 使用调度器提交任务
            try:
                scheduler_instance.submit_train(**train_kwargs)
            except Exception as e:
                logger.error(f"Experiment {run_label} failed: {e}")
                if not all_experiments and runs == 1:
                    raise  # 单次运行失败时直接抛出


@app.command()
def predict(
    pipeline_config: Annotated[str, typer.Option("-c", "--config", help="Pipeline 配置文件路径")],
    model_path: Annotated[str, typer.Option("-m", "--model", help="模型文件路径")],
    output_path: Annotated[
        Optional[str],
        typer.Option("-o", "--output", help="输出文件路径（支持 .root, .parquet, .json）"),
    ] = None,
    batch_size: Annotated[int, typer.Option("--batch-size", help="批次大小")] = 32,
    return_probabilities: Annotated[bool, typer.Option("--probabilities/--no-probabilities", help="是否返回概率")] = False,
    scheduler: Annotated[str, typer.Option("--scheduler", help="调度器类型：local 或 slurm")] = "local",
    slurm_config: Annotated[
        Optional[str],
        typer.Option("--slurm-config", help="SLURM 配置文件路径（仅 slurm 调度器）"),
    ] = None,
) -> None:
    """使用模型进行预测。

    输出格式：
    - .root: ROOT 文件，包含预测结果、标签和观察变量（类似 weaver）
    - .parquet: Parquet 文件，包含预测结果、标签和观察变量
    - .json: JSON 文件，包含预测结果（向后兼容）

    示例:
        bamboohepml predict -c configs/pipeline.yaml -m outputs/model.pt -o predictions.root
        bamboohepml predict -c configs/pipeline.yaml -m outputs/model.pt -o predictions.parquet
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
    model_path: Annotated[str, typer.Option("-m", "--model", help="模型文件路径")],
    output_path: Annotated[str, typer.Option("-o", "--output", help="输出 ONNX 文件路径")],
    metadata_path: Annotated[
        Optional[str],
        typer.Option(
            "--metadata",
            help="元数据文件路径（默认为 model_path 同目录下的 metadata.json）",
        ),
    ] = None,
    input_shape: Annotated[
        Optional[str],
        typer.Option(
            "--input-shape",
            help="输入形状，格式：'batch,features' 或 'features'（如果提供，将覆盖 metadata）",
        ),
    ] = None,
    opset_version: Annotated[int, typer.Option("--opset-version", help="ONNX opset 版本")] = 11,
    pipeline_config: Annotated[
        Optional[str],
        typer.Option("-c", "--config", help="Pipeline 配置文件路径（已废弃，向后兼容用）"),
    ] = None,
    scheduler: Annotated[str, typer.Option("--scheduler", help="调度器类型：local 或 slurm")] = "local",
    slurm_config: Annotated[
        Optional[str],
        typer.Option("--slurm-config", help="SLURM 配置文件路径（仅 slurm 调度器）"),
    ] = None,
) -> None:
    """导出模型为 ONNX 格式。

    导出不依赖 Dataset 和 Pipeline，只依赖模型和 metadata。

    示例:
        bamboohepml export -m outputs/model.pt -o model.onnx
        bamboohepml export -m outputs/model.pt -o model.onnx --metadata outputs/metadata.json
    """
    logger.info("=" * 80)
    logger.info("BambooHepMl Export Command")
    logger.info("=" * 80)

    # 验证文件存在
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
        "model_path": model_path,
        "output_path": output_path,
        "metadata_path": metadata_path,
        "input_shape": parsed_input_shape,
        "opset_version": opset_version,
        "pipeline_config_path": pipeline_config,  # 向后兼容
    }

    # 使用调度器提交任务
    scheduler_instance.submit_export(**export_kwargs)


@app.command()
def inspect(
    pipeline_config: Annotated[str, typer.Option("-c", "--config", help="Pipeline 配置文件路径")],
    output_path: Annotated[Optional[str], typer.Option("-o", "--output", help="输出文件路径")] = None,
    num_samples: Annotated[int, typer.Option("--num-samples", help="检查的样本数量")] = 1000,
    inspect_data: Annotated[bool, typer.Option("--inspect-data/--no-inspect-data", help="是否检查数据")] = True,
    inspect_features: Annotated[
        bool,
        typer.Option("--inspect-features/--no-inspect-features", help="是否检查特征"),
    ] = True,
    scheduler: Annotated[str, typer.Option("--scheduler", help="调度器类型：local 或 slurm")] = "local",
    slurm_config: Annotated[
        Optional[str],
        typer.Option("--slurm-config", help="SLURM 配置文件路径（仅 slurm 调度器）"),
    ] = None,
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
