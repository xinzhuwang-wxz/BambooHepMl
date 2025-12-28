"""
全局配置模块

借鉴 Made-With-ML 的配置管理方式，提供：
- 目录管理
- MLflow 配置
- 日志配置
"""
import logging
import logging.config
import os
import sys
from pathlib import Path

# MLflow 延迟导入（避免 protobuf 版本冲突）
_mlflow = None
try:
    import mlflow
    _mlflow = mlflow
except ImportError:
    # MLflow 不可用，但不影响基本功能
    pass

# 目录配置
ROOT_DIR = Path(__file__).parent.parent.absolute()
LOGS_DIR = Path(ROOT_DIR, "logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# 共享存储目录（用于 MLflow 和模型存储）
EFS_DIR = Path(f"/efs/shared_storage/bamboohepml/{os.environ.get('USER', 'default')}")
try:
    Path(EFS_DIR).mkdir(parents=True, exist_ok=True)
except OSError:
    EFS_DIR = Path(ROOT_DIR, "efs")
    Path(EFS_DIR).mkdir(parents=True, exist_ok=True)

# MLflow 配置（仅在可用时设置）
if _mlflow is not None:
    MODEL_REGISTRY = Path(f"{EFS_DIR}/mlflow")
    Path(MODEL_REGISTRY).mkdir(parents=True, exist_ok=True)
    MLFLOW_TRACKING_URI = "file://" + str(MODEL_REGISTRY.absolute())
    try:
        _mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    except Exception:
        # MLflow 配置失败，但不影响基本功能
        pass
else:
    MODEL_REGISTRY = Path(f"{EFS_DIR}/mlflow")
    MLFLOW_TRACKING_URI = None

# 日志配置
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "minimal": {"format": "%(message)s"},
        "detailed": {
            "format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.DEBUG,
        },
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "info.log"),
            "maxBytes": 10485760,  # 10 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "error.log"),
            "maxBytes": 10485760,  # 10 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.ERROR,
        },
    },
    "root": {
        "handlers": ["console", "info", "error"],
        "level": logging.INFO,
        "propagate": True,
    },
}

# 初始化日志
logging.config.dictConfig(logging_config)
logger = logging.getLogger()

# 导出常用配置
__all__ = ['logger', 'EFS_DIR', 'MLFLOW_TRACKING_URI', 'MODEL_REGISTRY']

