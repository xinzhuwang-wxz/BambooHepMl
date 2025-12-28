"""
Dataset 类

提供：
- 支持 jagged array（变长数组）
- 支持 padding + mask
- 支持 transformer 输入格式
- 与特征系统集成
"""

import copy
from functools import partial
from typing import Any, Dict, Optional

import awkward as ak
import numpy as np
import torch
from torch.utils.data import IterableDataset

from .config import DataConfig
from .logger import _logger
from .preprocess import _apply_selection, _build_new_variables, _build_weights
from .sources.base import DataSource
from .tools import _clip, _pad, _repeat_pad, _stack


def _collate_awkward_array_fn(batch, *, collate_fn_map=None):
    """Collate 函数：处理 awkward array。

    Args:
        batch: 批次数据
        collate_fn_map: collate 函数映射

    Returns:
        堆叠后的数组
    """
    return _stack(batch, axis=0)


def _finalize_inputs(table: ak.Array, data_config: DataConfig) -> Dict[str, Any]:
    """最终化输入（标准化、裁剪、padding、堆叠）。

    借鉴 weaver-core 的 _finalize_inputs 函数。

    Args:
        table: 数据表（awkward Array）
        data_config: 数据配置

    Returns:
        Dict[str, Any]: 处理后的输入字典
    """
    output = {}

    # 1. 复制观察者变量（在转换之前）
    for k in data_config.z_variables:
        if k in data_config.observer_names:
            output[k] = table[k]  # ak.Array

    # 2. 复制标签
    for k in data_config.label_names:
        output[k] = ak.to_numpy(table[k])

    # 3. 转换（标准化、裁剪、padding）
    for k, params in data_config.preprocess_params.items():
        if data_config._auto_standardization and params["center"] == "auto":
            raise ValueError(f"No valid standardization params for {k}")

        # 如果变量不在表中，尝试通过表达式构建（处理 Jet.pt 这种情况）
        if k not in table.fields:
            # 尝试通过表达式构建（如果包含点号，可能是属性访问）
            if "." in k:
                from bamboohepml.data.tools import _eval_expr

                try:
                    # 构建新变量并添加到表中
                    new_value = _eval_expr(k, table)
                    # 使用 ak.with_field 添加新字段
                    table = ak.with_field(table, new_value, k)
                except Exception as e:
                    _logger.warning(f"Could not build variable {k} from expression: {e}, skipping")
                    continue
            else:
                _logger.warning(f"Variable {k} not found in table, skipping")
                continue

        # 标准化和裁剪
        if params["center"] is not None:
            value = _clip((table[k] - params["center"]) * params["scale"], params["min"], params["max"])
            table = ak.with_field(table, value, k)

        # Padding（如果指定了长度）
        if params["length"] is not None:
            if params["pad_mode"] == "wrap":
                pad_fn = partial(_repeat_pad, shuffle=False)
            else:
                pad_fn = partial(_pad, value=params["pad_value"])
            value = pad_fn(table[k], params["length"])
            table = ak.with_field(table, value, k)

        # 检查 NaN
        current_value = table[k]
        if isinstance(current_value, ak.Array):
            numpy_k = ak.to_numpy(current_value)
            if np.any(np.isnan(numpy_k)):
                _logger.warning(f"Found NaN in {k}, silently converting it to 0.")
                value = np.nan_to_num(numpy_k)
                table = ak.with_field(table, value, k)
        elif isinstance(current_value, np.ndarray):
            if np.any(np.isnan(current_value)):
                _logger.warning(f"Found NaN in {k}, silently converting it to 0.")
                value = np.nan_to_num(current_value)
                table = ak.with_field(table, value, k)

    # 4. 堆叠变量（为每个输入组）
    for k, names in data_config.input_dicts.items():
        if len(names) == 0:
            # 空输入组，跳过
            continue
        elif len(names) == 1 and data_config.preprocess_params[names[0]]["length"] is None:
            # 单个变量，无 padding
            output["_" + k] = ak.to_numpy(ak.values_astype(table[names[0]], "float32"))
        else:
            # 多个变量，堆叠
            arrays_to_stack = [ak.to_numpy(table[n]).astype("float32") for n in names if n in table.fields]
            if len(arrays_to_stack) == 0:
                # 没有可堆叠的数组，跳过
                continue
            stacked = np.stack(arrays_to_stack, axis=1)
            output["_" + k] = ak.to_numpy(stacked)

    # 5. 复制监控变量（转换之后）
    for k in data_config.z_variables:
        if k in data_config.monitor_variables:
            output[k] = table[k]  # ak.Array

    return output


def _get_reweight_indices(weights, up_sample=True, max_resample=10, weight_scale=1):
    """获取重加权索引。

    借鉴 weaver-core 的实现。

    Args:
        weights: 权重数组
        up_sample: 是否上采样
        max_resample: 最大重采样次数
        weight_scale: 权重缩放因子

    Returns:
        np.ndarray: 重加权后的索引
    """
    all_indices = np.arange(len(weights))
    randwgt = np.random.uniform(low=0, high=weight_scale, size=len(weights))
    keep_flags = randwgt < weights

    if not up_sample:
        keep_indices = all_indices[keep_flags]
    else:
        n_repeats = len(weights) // max(1, int(keep_flags.sum()))
        if n_repeats > max_resample:
            n_repeats = max_resample
        all_indices = np.repeat(np.arange(len(weights)), n_repeats)
        randwgt = np.random.uniform(low=0, high=weight_scale, size=len(weights) * n_repeats)
        keep_indices = all_indices[randwgt < np.repeat(weights, n_repeats)]

    return copy.deepcopy(keep_indices)


def _check_labels(table: ak.Array, data_config: DataConfig):
    """检查标签一致性。

    Args:
        table: 数据表
        data_config: 数据配置
    """
    if "_labelcheck_" not in table.fields:
        return

    labelcheck = ak.to_numpy(table["_labelcheck_"])
    if np.all(labelcheck == 1):
        return
    else:
        if np.any(labelcheck == 0):
            raise RuntimeError("Inconsistent label definition: some entries are not assigned to any classes!")
        if np.any(labelcheck > 1):
            raise RuntimeError("Inconsistent label definition: some entries are assigned to multiple classes!")


class HEPDataset(IterableDataset):
    """
    HEP 数据集类

    支持：
    - Jagged array（变长数组）
    - Padding + Mask
    - Transformer 输入格式
    - 与特征系统集成
    """

    def __init__(
        self,
        data_source: DataSource,
        data_config: DataConfig,
        feature_graph=None,
        expression_engine=None,
        for_training: bool = True,
        shuffle: bool = True,
        reweight: bool = True,
        up_sample: bool = True,
        weight_scale: float = 1.0,
        max_resample: int = 10,
        selection: Optional[str] = None,
    ):
        """初始化数据集。

        Args:
            data_source: 数据源
            data_config: 数据配置
            feature_graph: 特征依赖图（可选）
            expression_engine: 表达式引擎（可选）
            for_training: 是否用于训练。默认为 True。
            shuffle: 是否打乱。默认为 True。
            reweight: 是否重加权。默认为 True。
            up_sample: 是否上采样。默认为 True。
            weight_scale: 权重缩放因子。默认为 1.0。
            max_resample: 最大重采样次数。默认为 10。
            selection: 额外的选择条件。默认为 None。
        """
        self.data_source = data_source
        self.data_config = data_config
        self.feature_graph = feature_graph
        self.expression_engine = expression_engine
        self.for_training = for_training
        self.shuffle = shuffle if for_training else False
        self.reweight = reweight if for_training else False
        self.up_sample = up_sample
        self.weight_scale = weight_scale
        self.max_resample = max_resample
        self.extra_selection = selection

        # 注册 collate 函数
        from torch.utils.data._utils.collate import default_collate_fn_map

        default_collate_fn_map.update({ak.Array: _collate_awkward_array_fn})

    def _load_and_preprocess(self) -> tuple:
        """加载并预处理数据。

        Returns:
            tuple: (table, indices) - 处理后的数据表和索引
        """
        # 1. 确定要加载的分支
        load_branches = self.data_config.train_load_branches if self.for_training else self.data_config.test_load_branches

        # 2. 加载数据
        table = self.data_source.load_branches(list(load_branches))

        # 3. 应用选择条件
        selection = self.data_config.selection if self.for_training else self.data_config.test_time_selection
        if self.extra_selection:
            if selection:
                selection = f"({selection}) & ({self.extra_selection})"
            else:
                selection = self.extra_selection

        table = _apply_selection(table, selection, funcs=self.data_config.var_funcs)

        if len(table) == 0:
            return [], []

        # 4. 构建新变量
        aux_branches = self.data_config.train_aux_branches if self.for_training else self.data_config.test_aux_branches
        table = _build_new_variables(table, {k: v for k, v in self.data_config.var_funcs.items() if k in aux_branches})

        # 5. 检查标签
        if self.data_config.label_type == "simple" and self.for_training:
            _check_labels(table, self.data_config)

        # 6. 计算重加权索引
        if self.reweight and self.data_config.weight_name is not None:
            weights = _build_weights(table, self.data_config)
            indices = _get_reweight_indices(weights, up_sample=self.up_sample, weight_scale=self.weight_scale, max_resample=self.max_resample)
        else:
            indices = np.arange(len(table[self.data_config.label_names[0]]))

        # 7. 打乱
        if self.shuffle:
            np.random.shuffle(indices)

        # 8. 应用特征系统（如果提供，在最终化之前）
        if self.feature_graph is not None and self.expression_engine is not None:
            table = self._apply_feature_system(table)

        # 9. 最终化输入（标准化、裁剪、padding、堆叠）
        table = _finalize_inputs(table, self.data_config)

        return table, indices

    def _apply_feature_system(self, table: ak.Array) -> ak.Array:
        """应用特征系统（如果提供了 feature_graph）。

        Args:
            table: 原始数据表

        Returns:
            ak.Array: 应用特征后的数据表
        """
        if self.feature_graph is None or self.expression_engine is None:
            return table

        # 构建上下文
        context = {k: table[k] for k in table.fields}

        # 按执行顺序计算特征
        execution_order = self.feature_graph.get_execution_order()

        for feature_name in execution_order:
            # 检查缓存
            cached_value = self.feature_graph.get_cached_value(feature_name)
            if cached_value is not None:
                context[feature_name] = cached_value
                table[feature_name] = cached_value
                continue

            # 计算特征
            feature_def = self.feature_graph.nodes[feature_name].feature_def

            try:
                if "expr" in feature_def:
                    raw_value = self.expression_engine.evaluate(feature_def["expr"], context)
                else:
                    source = feature_def.get("source")
                    if isinstance(source, list):
                        # 如果是列表，使用第一个
                        source = source[0]
                    raw_value = context.get(source)

                # 处理特征（标准化、裁剪、padding）
                from .features.processors import FeatureProcessor

                processor = FeatureProcessor(feature_def)
                processed_value = processor.process(raw_value, fit_normalizer=True)

                # 缓存
                self.feature_graph.cache_value(feature_name, processed_value)
                context[feature_name] = processed_value
                table[feature_name] = processed_value
            except Exception as e:
                _logger.warning(f"Failed to compute feature '{feature_name}': {e}")
                continue

        return table

    def __iter__(self):
        """迭代器：返回数据批次。

        Yields:
            Dict[str, Any]: 数据批次
        """
        # 加载并预处理
        table, indices = self._load_and_preprocess()

        if len(indices) == 0:
            return

        # 应用特征系统（如果提供）
        # 注意：特征系统应该在预处理之后应用，因为需要原始数据作为输入
        # 但这里我们在 _finalize_inputs 之前应用，以便特征可以参与最终化

        # 按索引返回数据
        for idx in indices:
            sample = {}
            for key, value in table.items():
                try:
                    if isinstance(value, ak.Array):
                        sample[key] = value[idx]
                    elif isinstance(value, np.ndarray):
                        # 转换为 torch.Tensor（DataLoader 会自动批处理）
                        item = value[idx]
                        # 处理标量值（numpy.int64, numpy.float64 等）
                        if np.isscalar(item):
                            sample[key] = torch.tensor(item)
                        else:
                            sample[key] = torch.from_numpy(item)
                    else:
                        sample[key] = value
                except (IndexError, KeyError):
                    # 跳过无法访问的键
                    continue

            yield sample

    def __len__(self) -> int:
        """获取数据集长度。

        Returns:
            int: 数据集长度
        """
        num_events = self.data_source.get_num_events()
        if num_events is not None:
            return num_events
        else:
            # 如果无法确定，返回一个估计值
            return 0

    def get_sample(self, index: int) -> Dict[str, Any]:
        """获取单个样本（用于调试）。

        Args:
            index: 样本索引

        Returns:
            Dict[str, Any]: 样本数据
        """
        table, indices = self._load_and_preprocess()

        if index >= len(indices):
            raise IndexError(f"Index {index} out of range")

        idx = indices[index]
        sample = {}
        for key, value in table.items():
            if isinstance(value, ak.Array):
                sample[key] = value[idx]
            elif isinstance(value, np.ndarray):
                sample[key] = value[idx]
            else:
                sample[key] = value

        return sample


class TransformerDataset(HEPDataset):
    """
    Transformer 输入格式数据集

    专门为 Transformer 模型设计，输出格式：
    - x: (N, C, P) - 特征序列
    - mask: (N, P) - 注意力掩码（True 表示有效，False 表示 padding）
    - v: (N, 4, P) - 四动量（可选）
    """

    def __init__(self, *args, **kwargs):
        """初始化 Transformer 数据集。"""
        super().__init__(*args, **kwargs)

    def _format_transformer_input(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """格式化为 Transformer 输入格式。

        Args:
            sample: 原始样本

        Returns:
            Dict[str, torch.Tensor]: Transformer 格式的输入
        """
        output = {}

        # 提取输入组
        for input_name in self.data_config.input_names:
            input_key = "_" + input_name

            if input_key not in sample:
                continue

            # 获取输入数据
            input_data = sample[input_key]

            # 转换为 numpy 数组
            if isinstance(input_data, ak.Array):
                input_data = ak.to_numpy(input_data)

            # 转换为 torch tensor
            # 假设形状为 (P, C) 或 (C, P)，需要转换为 (C, P)
            if input_data.ndim == 2:
                if input_data.shape[0] > input_data.shape[1]:
                    # (P, C) -> (C, P)
                    input_data = input_data.T
                # 添加 batch 维度: (C, P) -> (1, C, P)
                input_data = input_data[np.newaxis, :, :]

            output["x"] = torch.from_numpy(input_data.astype(np.float32))

            # 生成 mask（如果有 padding）
            if input_name.endswith("_mask"):
                # 使用专门的 mask 特征
                mask_key = input_name
                if mask_key in sample:
                    mask_data = sample[mask_key]
                    if isinstance(mask_data, ak.Array):
                        mask_data = ak.to_numpy(mask_data)
                    output["mask"] = torch.from_numpy(mask_data.astype(bool))
                else:
                    # 如果没有专门的 mask，从输入数据生成
                    # 假设 padding 值为 0
                    mask = (input_data != 0).any(axis=1)  # (1, P)
                    output["mask"] = torch.from_numpy(mask.astype(bool))
            else:
                # 从输入数据生成 mask（假设 padding 为 0）
                mask = (input_data != 0).any(axis=1)  # (1, P)
                output["mask"] = torch.from_numpy(mask.astype(bool))

            # 提取四动量（如果存在）
            if "v" in sample:
                v_data = sample["v"]
                if isinstance(v_data, ak.Array):
                    v_data = ak.to_numpy(v_data)
                if v_data.ndim == 2:
                    v_data = v_data.T[np.newaxis, :, :]  # (1, 4, P)
                output["v"] = torch.from_numpy(v_data.astype(np.float32))

            break  # 只处理第一个输入组

        # 添加标签
        for label_name in self.data_config.label_names:
            if label_name in sample:
                label_data = sample[label_name]
                if isinstance(label_data, ak.Array):
                    label_data = ak.to_numpy(label_data)
                output[label_name] = torch.from_numpy(label_data.astype(np.int64))

        return output

    def __iter__(self):
        """迭代器：返回 Transformer 格式的数据批次。

        Yields:
            Dict[str, torch.Tensor]: Transformer 格式的数据
        """
        for sample in super().__iter__():
            yield self._format_transformer_input(sample)
