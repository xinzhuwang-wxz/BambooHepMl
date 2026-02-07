"""
SLURM 调度器

使用 SLURM 提交任务到集群。
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from ..config import logger
from .base import BaseScheduler


class SLURMScheduler(BaseScheduler):
    """
    SLURM 调度器

    使用 SLURM 提交任务到集群。
    """

    def __init__(
        self,
        slurm_config_path: Optional[str] = None,
        default_partition: str = "gpu",
        default_gpus: int = 1,
        default_cpus: int = 4,
        default_memory: str = "16G",
        default_time: str = "24:00:00",
    ):
        """
        初始化 SLURM 调度器。

        Args:
            slurm_config_path: SLURM 配置文件路径（可选）
            default_partition: 默认分区
            default_gpus: 默认 GPU 数量
            default_cpus: 默认 CPU 数量
            default_memory: 默认内存
            default_time: 默认时间限制
        """
        self.slurm_config_path = slurm_config_path
        self.default_partition = default_partition
        self.default_gpus = default_gpus
        self.default_cpus = default_cpus
        self.default_memory = default_memory
        self.default_time = default_time

        # 加载 SLURM 配置（如果提供）
        self.slurm_config = {}
        if slurm_config_path and Path(slurm_config_path).exists():
            self._load_slurm_config()

    def _load_slurm_config(self):
        """加载 SLURM 配置文件。"""
        # 简化实现：可以扩展为解析 YAML/JSON 配置文件
        logger.info(f"Loading SLURM config from {self.slurm_config_path}")
        # 这里可以添加配置解析逻辑

    def _generate_sbatch_script(
        self,
        command: str,
        job_name: str,
        partition: Optional[str] = None,
        gpus: Optional[int] = None,
        cpus: Optional[int] = None,
        memory: Optional[str] = None,
        time: Optional[str] = None,
        output: Optional[str] = None,
        error: Optional[str] = None,
    ) -> str:
        """
        生成 sbatch 脚本。

        Args:
            command: 要执行的命令
            job_name: 作业名称
            partition: 分区（如果为 None，使用默认值）
            gpus: GPU 数量（如果为 None，使用默认值）
            cpus: CPU 数量（如果为 None，使用默认值）
            memory: 内存（如果为 None，使用默认值）
            time: 时间限制（如果为 None，使用默认值）
            output: 输出文件路径（如果为 None，自动生成）
            error: 错误文件路径（如果为 None，自动生成）

        Returns:
            sbatch 脚本内容
        """
        partition = partition or self.default_partition
        gpus = gpus or self.default_gpus
        cpus = cpus or self.default_cpus
        memory = memory or self.default_memory
        time = time or self.default_time

        if output is None:
            output = f"slurm_outputs/{job_name}_%j.out"
        if error is None:
            error = f"slurm_outputs/{job_name}_%j.err"

        script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{gpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={memory}
#SBATCH --time={time}
#SBATCH --output={output}
#SBATCH --error={error}

# 设置环境
source ~/.bashrc
conda activate bamboohepml  # 根据需要修改

# 执行命令
{command}
"""
        return script

    def _submit_sbatch(self, script_path: str) -> str:
        """
        提交 sbatch 脚本。

        Args:
            script_path: 脚本路径

        Returns:
            作业 ID
        """
        try:
            result = subprocess.run(
                ["sbatch", script_path],
                capture_output=True,
                text=True,
                check=True,
            )
            # 解析作业 ID（sbatch 输出格式：Submitted batch job 12345）
            job_id = result.stdout.strip().split()[-1]
            logger.info(f"Submitted SLURM job: {job_id}")
            return job_id
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to submit SLURM job: {e.stderr}")
            raise

    def submit_train(
        self,
        pipeline_config_path: str,
        experiment_name: Optional[str] = None,
        num_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        output_dir: Optional[str] = None,
        use_ray: bool = False,
        num_workers: int = 1,
        gpu_per_worker: int = 0,
        task_type: Optional[str] = None,
        model_type: Optional[str] = None,
        run_index: int = 1,
        **kwargs,
    ) -> str:
        """提交训练任务到 SLURM。"""
        logger.info("Using SLURM Scheduler for training")

        # 构建命令
        cmd_parts = [
            "bamboohepml train",
            f"-c {pipeline_config_path}",
        ]

        if experiment_name:
            cmd_parts.append(f"--experiment-name {experiment_name}")
        if num_epochs:
            cmd_parts.append(f"--num-epochs {num_epochs}")
        if batch_size:
            cmd_parts.append(f"--batch-size {batch_size}")
        if learning_rate:
            cmd_parts.append(f"--learning-rate {learning_rate}")
        if output_dir:
            cmd_parts.append(f"--output-dir {output_dir}")
        if use_ray:
            cmd_parts.append("--use-ray")
            cmd_parts.append(f"--num-workers {num_workers}")
            cmd_parts.append(f"--gpu-per-worker {gpu_per_worker}")

        # Forward multi-experiment params as single --task/--model
        # (the CLI loop already resolved --all/--runs into individual calls)
        if task_type:
            cmd_parts.append(f"--task {task_type}")
        if model_type:
            cmd_parts.append(f"--model {model_type}")

        cmd_parts.append("--scheduler local")  # 在 SLURM 作业中使用 local scheduler

        command = " ".join(cmd_parts)

        # 确定 GPU 数量
        gpus = gpu_per_worker * num_workers if use_ray else (gpu_per_worker or self.default_gpus)

        # 生成作业名称
        name_parts = ["train"]
        if task_type:
            name_parts.append(task_type)
        if model_type:
            name_parts.append(model_type)
        name_parts.append(experiment_name or "default")
        if run_index > 1:
            name_parts.append(f"run{run_index}")
        job_name = "_".join(name_parts)

        # 生成 sbatch 脚本
        script = self._generate_sbatch_script(
            command=command,
            job_name=job_name,
            gpus=gpus,
        )

        # 保存脚本并提交
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            job_id = self._submit_sbatch(script_path)
            logger.info(f"Training job submitted: {job_id}")
            return job_id
        finally:
            # 不删除脚本，保留用于调试
            pass

    def submit_predict(
        self,
        pipeline_config_path: str,
        model_path: str,
        output_path: Optional[str] = None,
        batch_size: int = 32,
        return_probabilities: bool = False,
        **kwargs,
    ) -> str:
        """提交预测任务到 SLURM。"""
        logger.info("Using SLURM Scheduler for prediction")

        # 构建命令
        cmd_parts = [
            "bamboohepml predict",
            f"-c {pipeline_config_path}",
            f"-m {model_path}",
            f"--batch-size {batch_size}",
        ]

        if output_path:
            cmd_parts.append(f"-o {output_path}")
        if return_probabilities:
            cmd_parts.append("--probabilities")

        cmd_parts.append("--scheduler local")

        command = " ".join(cmd_parts)

        # 生成作业名称
        job_name = "predict"

        # 生成 sbatch 脚本（预测通常不需要 GPU）
        script = self._generate_sbatch_script(
            command=command,
            job_name=job_name,
            gpus=0,  # 预测可能不需要 GPU
        )

        # 保存脚本并提交
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            job_id = self._submit_sbatch(script_path)
            logger.info(f"Prediction job submitted: {job_id}")
            return job_id
        finally:
            pass

    def submit_export(
        self,
        model_path: str,
        output_path: str,
        metadata_path: Optional[str] = None,
        input_shape: Optional[tuple] = None,
        opset_version: int = 11,
        pipeline_config_path: Optional[str] = None,  # 向后兼容，已废弃
        **kwargs,
    ) -> str:
        """提交导出任务到 SLURM。"""
        logger.info("Using SLURM Scheduler for export")

        # 构建命令
        cmd_parts = [
            "bamboohepml export",
            f"-m {model_path}",
            f"-o {output_path}",
            f"--opset-version {opset_version}",
        ]

        if metadata_path:
            cmd_parts.append(f"--metadata {metadata_path}")

        if input_shape:
            input_shape_str = ",".join(map(str, input_shape))
            cmd_parts.append(f"--input-shape {input_shape_str}")

        if pipeline_config_path:  # 向后兼容
            cmd_parts.append(f"-c {pipeline_config_path}")

        cmd_parts.append("--scheduler local")

        command = " ".join(cmd_parts)

        # 生成作业名称
        job_name = "export"

        # 生成 sbatch 脚本（导出通常不需要 GPU）
        script = self._generate_sbatch_script(
            command=command,
            job_name=job_name,
            gpus=0,
        )

        # 保存脚本并提交
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            job_id = self._submit_sbatch(script_path)
            logger.info(f"Export job submitted: {job_id}")
            return job_id
        finally:
            pass

    def submit_inspect(
        self,
        pipeline_config_path: str,
        output_path: Optional[str] = None,
        num_samples: int = 1000,
        inspect_data: bool = True,
        inspect_features: bool = True,
        **kwargs,
    ) -> str:
        """提交检查任务到 SLURM。"""
        logger.info("Using SLURM Scheduler for inspection")

        # 构建命令
        cmd_parts = [
            "bamboohepml inspect",
            f"-c {pipeline_config_path}",
            f"--num-samples {num_samples}",
        ]

        if output_path:
            cmd_parts.append(f"-o {output_path}")
        if not inspect_data:
            cmd_parts.append("--no-inspect-data")
        if not inspect_features:
            cmd_parts.append("--no-inspect-features")

        cmd_parts.append("--scheduler local")

        command = " ".join(cmd_parts)

        # 生成作业名称
        job_name = "inspect"

        # 生成 sbatch 脚本（检查通常不需要 GPU）
        script = self._generate_sbatch_script(
            command=command,
            job_name=job_name,
            gpus=0,
        )

        # 保存脚本并提交
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            job_id = self._submit_sbatch(script_path)
            logger.info(f"Inspection job submitted: {job_id}")
            return job_id
        finally:
            pass
