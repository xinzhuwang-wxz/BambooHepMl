"""
特征处理器模块

提供：
- FeatureProcessor: 统一特征处理器
- Normalizer: 标准化处理器
- Clipper: 裁剪处理器
- Padder: 填充处理器
"""

from typing import Any, Optional, Union

import awkward as ak
import numpy as np

from ..logger import _logger


class Normalizer:
    """标准化处理器。"""

    def __init__(self, method: str = "auto", center: Optional[float] = None, scale: Optional[float] = None):
        """初始化标准化器。

        Args:
            method (str): 标准化方法（'auto', 'manual', 'none'）。默认为 'auto'。
            center (float, optional): 中心值（manual 模式）。默认为 None。
            scale (float, optional): 缩放因子（manual 模式）。默认为 None。
        """
        self.method = method
        self.center = center
        self.scale = scale
        self._fitted = False

    def fit(self, data: Union[np.ndarray, ak.Array]) -> "Normalizer":
        """拟合标准化参数（仅用于 auto 模式）。

        Args:
            data: 输入数据

        Returns:
            Normalizer: self
        """
        if self.method == "auto":
            # 展平数据（如果是 awkward array）
            if isinstance(data, ak.Array):
                flat_data = ak.to_numpy(ak.flatten(data, axis=None))
            else:
                flat_data = data.flatten()

            # 移除 NaN 和 Inf
            flat_data = flat_data[np.isfinite(flat_data)]

            if len(flat_data) == 0:
                _logger.warning("No valid data for normalization, using default values")
                self.center = 0.0
                self.scale = 1.0
            else:
                # 使用百分位数方法（16th, 50th, 84th）
                low, center, high = np.percentile(flat_data, [16, 50, 84])
                scale = max(high - center, center - low)
                scale = 1.0 if scale == 0 else 1.0 / scale

                self.center = float(center)
                self.scale = float(scale)

            self._fitted = True
            _logger.debug(f"Fitted normalizer: center={self.center}, scale={self.scale}")

        return self

    def transform(self, data: Union[np.ndarray, ak.Array]) -> Union[np.ndarray, ak.Array]:
        """应用标准化。

        Args:
            data: 输入数据

        Returns:
            标准化后的数据
        """
        if self.method == "none" or (self.center is None and self.scale is None):
            return data

        if self.method == "auto" and not self._fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        center = self.center if self.center is not None else 0.0
        scale = self.scale if self.scale is not None else 1.0

        if isinstance(data, ak.Array):
            return (data - center) * scale
        else:
            return (data - center) * scale

    def fit_transform(self, data: Union[np.ndarray, ak.Array]) -> Union[np.ndarray, ak.Array]:
        """拟合并转换。

        Args:
            data: 输入数据

        Returns:
            标准化后的数据
        """
        return self.fit(data).transform(data)


class Clipper:
    """裁剪处理器。"""

    def __init__(self, min_val: Optional[float] = None, max_val: Optional[float] = None):
        """初始化裁剪器。

        Args:
            min_val (float, optional): 最小值。默认为 None（不裁剪下界）。
            max_val (float, optional): 最大值。默认为 None（不裁剪上界）。
        """
        self.min_val = min_val
        self.max_val = max_val

    def clip(self, data: Union[np.ndarray, ak.Array]) -> Union[np.ndarray, ak.Array]:
        """裁剪数据。

        Args:
            data: 输入数据

        Returns:
            裁剪后的数据
        """
        if self.min_val is None and self.max_val is None:
            return data

        if isinstance(data, ak.Array):
            # Convert to numpy for clipping, then back to awkward
            numpy_data = ak.to_numpy(data)
            clipped = np.clip(numpy_data, self.min_val, self.max_val)
            # Try to preserve structure if possible
            if data.ndim == 1:
                return ak.Array(clipped)
            else:
                return ak.from_numpy(clipped)
        else:
            return np.clip(data, self.min_val, self.max_val)


class Padder:
    """填充处理器。"""

    def __init__(self, max_length: int, mode: str = "constant", value: float = 0.0):
        """初始化填充器。

        Args:
            max_length (int): 最大长度
            mode (str): 填充模式（'constant', 'wrap', 'repeat'）。默认为 'constant'。
            value (float): 填充值（constant 模式）。默认为 0.0。
        """
        self.max_length = max_length
        self.mode = mode
        self.value = value

    def pad(self, data: ak.Array) -> np.ndarray:
        """填充数据。

        Args:
            data: 输入数据（必须是 awkward Array）

        Returns:
            填充后的 numpy 数组
        """
        if not isinstance(data, ak.Array):
            raise ValueError("Padder only works with awkward Arrays")

        if self.mode == "constant":
            return self._pad_constant(data)
        elif self.mode == "wrap":
            return self._pad_wrap(data)
        elif self.mode == "repeat":
            return self._pad_repeat(data)
        else:
            raise ValueError(f"Unknown padding mode: {self.mode}")

    def _pad_constant(self, data: ak.Array) -> np.ndarray:
        """常量填充。"""
        if data.ndim == 1:
            data = ak.unflatten(data, 1)
        padded = ak.fill_none(ak.pad_none(data, self.max_length, clip=True), self.value)
        return ak.to_numpy(padded)

    def _pad_wrap(self, data: ak.Array) -> np.ndarray:
        """循环填充。"""
        # 简化实现：重复数据直到达到 max_length
        return self._pad_repeat(data)

    def _pad_repeat(self, data: ak.Array) -> np.ndarray:
        """重复填充。"""
        # 展平所有数据
        flat_data = ak.to_numpy(ak.flatten(data))

        # 计算需要重复的次数
        total_needed = len(data) * self.max_length
        repeat_times = int(np.ceil(total_needed / len(flat_data)))

        # 重复数据
        repeated = np.tile(flat_data, repeat_times)[:total_needed]

        # 重塑为 (n_events, max_length)
        return repeated.reshape((len(data), self.max_length))


class FeatureProcessor:
    """统一特征处理器。

    支持：
    - 标准化（normalize）
    - 裁剪（clip）
    - 填充（padding，仅 object-level）
    - 类型转换
    """

    def __init__(self, feature_def: dict):
        """初始化特征处理器。

        Args:
            feature_def (dict): 特征定义
        """
        self.feature_def = feature_def
        self.dtype = feature_def.get("dtype", "float32")
        self.feature_type = feature_def.get("type", "event")  # 'event' or 'object'

        # 初始化子处理器
        self.normalizer = None
        self.clipper = None
        self.padder = None

        # 标准化
        if "normalize" in feature_def:
            norm_config = feature_def["normalize"]
            method = norm_config.get("method", "auto")
            center = norm_config.get("center")
            scale = norm_config.get("scale")
            self.normalizer = Normalizer(method=method, center=center, scale=scale)

        # 裁剪
        if "clip" in feature_def:
            clip_config = feature_def["clip"]
            self.clipper = Clipper(min_val=clip_config.get("min"), max_val=clip_config.get("max"))

        # 填充（仅 object-level）
        if self.feature_type == "object" and "padding" in feature_def:
            pad_config = feature_def["padding"]
            self.padder = Padder(
                max_length=pad_config.get("max_length", 128),
                mode=pad_config.get("mode", "constant"),
                value=pad_config.get("value", 0.0),
            )

    def process(self, value: Any, fit_normalizer: bool = False) -> Any:
        """处理特征值。

        Args:
            value: 原始特征值
            fit_normalizer (bool): 是否拟合标准化器（仅第一次）。默认为 False。

        Returns:
            处理后的特征值
        """
        # 1. 标准化
        if self.normalizer:
            if fit_normalizer and self.normalizer.method == "auto":
                value = self.normalizer.fit_transform(value)
            else:
                value = self.normalizer.transform(value)

        # 2. 裁剪
        if self.clipper:
            value = self.clipper.clip(value)

        # 3. 类型转换
        if isinstance(value, ak.Array):
            value = ak.values_astype(value, self.dtype)
        else:
            value = np.asarray(value, dtype=self.dtype)

        # 4. 填充（仅 object-level）
        if self.padder and isinstance(value, ak.Array):
            value = self.padder.pad(value)

        return value
