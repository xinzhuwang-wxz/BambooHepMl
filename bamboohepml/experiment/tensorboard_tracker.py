"""
TensorBoard 跟踪器

提供 TensorBoard 专用的功能。
"""

from pathlib import Path


class TensorBoardTracker:
    """
    TensorBoard 跟踪器

    提供 TensorBoard 相关的工具函数。
    """

    def __init__(self, log_dir: str = "./logs/tensorboard"):
        """
        初始化 TensorBoard 跟踪器。

        Args:
            log_dir: 日志目录
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def get_log_dir(self) -> Path:
        """获取日志目录。"""
        return self.log_dir

    def start_tensorboard_command(self, port: int = 6006) -> str:
        """
        生成启动 TensorBoard 的命令。

        Args:
            port: 端口号

        Returns:
            启动命令字符串
        """
        return f"tensorboard --logdir {self.log_dir} --port {port} --host 0.0.0.0"
