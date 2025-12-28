"""
简化的日志模块（借鉴 weaver-core）
"""
import logging
import sys

# 尝试导入全局 logger，如果失败则创建本地 logger
try:
    from bamboohepml.config import logger as base_logger

    _logger = base_logger
except (ImportError, AttributeError):
    # 如果导入失败，创建本地 logger
    _logger = logging.getLogger("bamboohepml.data")
    if not _logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        _logger.addHandler(handler)
        _logger.setLevel(logging.INFO)

_warning_counter = {}


def warn_n_times(msg, n=10, logger=_logger):
    """限制警告消息的显示次数。

    Args:
        msg (str): 警告消息。
        n (int): 最多显示次数。默认为 10。
        logger: 日志记录器。
    """
    if msg not in _warning_counter:
        _warning_counter[msg] = 0
    if _warning_counter[msg] < n:
        logger.warning(msg)
    _warning_counter[msg] += 1
