"""
Dataset 类

提供：
- 支持 jagged array（变长数组）
- 支持 padding + mask
- 支持 transformer 输入格式
- 与特征系统集成
"""

from __future__ import annotations

import copy
from typing import Any

import awkward as ak
import numpy as np
import torch
from torch.utils.data import IterableDataset

from .config import DataConfig
from .preprocess import _apply_selection, _build_new_variables, _build_weights
from .sources.base import DataSource
from .tools import _stack


def _collate_awkward_array_fn(batch, *, collate_fn_map=None):
    """Collate 函数：处理 awkward array。

    Args:
        batch: 批次数据
        collate_fn_map: collate 函数映射

    Returns:
        堆叠后的数组
    """
    return _stack(batch, axis=0)


# _finalize_inputs 已废弃
# _apply_feature_system 已废弃


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

    注意：此数据集将所有请求的数据加载到内存中。对于大型数据集，请确保 `load_range` 设置适当，
    或使用能够分块流式传输的 DataSource（目前 ROOTDataSource 按文件加载全部指定范围）。
    为了提高性能，数据在首次加载后会被缓存。

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
        feature_graph,  # 现在是必需的
        for_training: bool = True,
        shuffle: bool = True,
        reweight: bool = True,
        up_sample: bool = True,
        weight_scale: float = 1.0,
        max_resample: int = 10,
        selection: str | None = None,
    ):
        """初始化数据集。

        Args:
            data_source: 数据源
            data_config: 数据配置（只包含数据源、selection、weights、labels）
            feature_graph: 特征依赖图（必需）- 现在 FeatureGraph 是唯一可信的特征事实源
            for_training: 是否用于训练。默认为 True。
            shuffle: 是否打乱。默认为 True。
            reweight: 是否重加权。默认为 True。
            up_sample: 是否上采样。默认为 True。
            weight_scale: 权重缩放因子。默认为 1.0。
            max_resample: 最大重采样次数。默认为 10。
            selection: 额外的选择条件。默认为 None。
        """
        if feature_graph is None:
            raise ValueError("feature_graph is required. Feature definitions must be in FeatureGraph, not DataConfig.")

        self.data_source = data_source
        self.data_config = data_config
        self.feature_graph = feature_graph
        self.for_training = for_training
        self.shuffle = shuffle if for_training else False
        self.reweight = reweight if for_training else False
        self.up_sample = up_sample
        self.weight_scale = weight_scale
        self.max_resample = max_resample
        self.extra_selection = selection

        # 内部缓存
        self._cached_batch = None
        self._cached_weights = None
        self._num_events = 0

        # 注册 collate 函数
        from torch.utils.data._utils.collate import default_collate_fn_map

        default_collate_fn_map.update({ak.Array: _collate_awkward_array_fn})

    def _load_and_preprocess(self) -> tuple:
        """加载并预处理数据。

        Returns:
            tuple: (batch, indices) - 处理后的批次（包含特征张量）和索引
        """
        # 检查缓存
        if self._cached_batch is not None:
            # 使用缓存数据重新生成索引（支持每个 epoch 重新采样/打乱）
            if self.reweight and self._cached_weights is not None:
                indices = _get_reweight_indices(
                    self._cached_weights, up_sample=self.up_sample, weight_scale=self.weight_scale, max_resample=self.max_resample
                )
            else:
                indices = np.arange(self._num_events)

            if self.shuffle:
                np.random.shuffle(indices)

            return self._cached_batch, indices

        # 1. 确定要加载的分支
        load_branches = self.data_config.train_load_branches if self.for_training else self.data_config.test_load_branches

        # 过滤掉计算字段（只保留原始字段）
        # 计算字段在 aux_branches 中，不应该被加载
        aux_branches = self.data_config.train_aux_branches if self.for_training else self.data_config.test_aux_branches
        if load_branches:
            load_branches = {b for b in load_branches if b not in aux_branches}

        # 2. 从 FeatureGraph 提取需要的数据源字段（始终执行，因为 FeatureGraph 是唯一特征源）
        feature_graph_fields = set()
        if self.feature_graph.expression_engine is not None:
            for node in self.feature_graph.nodes.values():
                feature_def = node.feature_def
                # 从 expr 提取依赖
                if "expr" in feature_def:
                    expr = feature_def["expr"]
                    try:
                        deps = self.feature_graph.expression_engine.get_dependencies(expr)
                        # 只保留不在特征图中的依赖（即原始数据字段）
                        for dep in deps:
                            if dep not in self.feature_graph.nodes:
                                feature_graph_fields.add(dep)
                    except Exception:
                        pass
                # 从 source 提取
                source = feature_def.get("source")
                if source and source not in self.feature_graph.nodes:
                    feature_graph_fields.add(source)

        # 3. 确保 load_branches 是 set 类型并合并 FeatureGraph 字段
        if isinstance(load_branches, set):
            load_branches = load_branches.copy()
        else:
            load_branches = set(load_branches) if load_branches else set()

        # 合并从 FeatureGraph 提取的字段
        load_branches |= feature_graph_fields

        # 4. 添加标签字段的原始数据源（如果存在）
        # label_value 中的字段（如 is_signal）是原始字段，需要被加载
        if self.data_config.label_value:
            if isinstance(self.data_config.label_value, list):
                # simple label type: label_value is a list like ["is_signal"]
                load_branches |= set(self.data_config.label_value)
            elif isinstance(self.data_config.label_value, dict):
                # complex label type: label_value is a dict
                load_branches |= set(self.data_config.label_value.keys())

        table = self.data_source.load_branches(list(load_branches))

        # 4. 应用选择条件
        selection = self.data_config.selection if self.for_training else self.data_config.test_time_selection
        if self.extra_selection:
            if selection:
                selection = f"({selection}) & ({self.extra_selection})"
            else:
                selection = self.extra_selection

        table = _apply_selection(table, selection, funcs=self.data_config.var_funcs)

        if len(table) == 0:
            return {}, []  # 返回空批次字典和空索引列表

        # 5. 构建新变量
        aux_branches = self.data_config.train_aux_branches if self.for_training else self.data_config.test_aux_branches
        table = _build_new_variables(table, {k: v for k, v in self.data_config.var_funcs.items() if k in aux_branches})

        # 6. 检查标签
        if self.data_config.label_type == "simple" and self.for_training:
            _check_labels(table, self.data_config)

        # 7. 计算重加权索引
        weights = None
        if self.reweight and self.data_config.weight_name is not None:
            weights = _build_weights(table, self.data_config)
            indices = _get_reweight_indices(weights, up_sample=self.up_sample, weight_scale=self.weight_scale, max_resample=self.max_resample)
        else:
            indices = np.arange(len(table))

        # 8. 打乱
        if self.shuffle:
            np.random.shuffle(indices)

        # 9. 使用 FeatureGraph 构建模型输入批次
        # FeatureGraph.build_batch() 会：
        # - 按执行顺序计算所有特征
        # - 应用预处理（normalize/clip/pad）
        # - 返回 torch.Tensor 字典（event/object/mask）
        feature_batch = self.feature_graph.build_batch(table)

        # 10. 添加标签和观察者变量
        batch = {}
        batch.update(feature_batch)  # event, object, mask

        # 添加标签
        for label_name in self.data_config.label_names:
            if label_name in table.fields:
                labels = ak.to_numpy(table[label_name])
                batch["_label_"] = torch.from_numpy(labels.astype(np.int64))

        # 添加观察者变量
        for obs_name in self.data_config.z_variables:
            if obs_name in table.fields:
                batch[obs_name] = table[obs_name]  # 保持为 ak.Array

        # 缓存结果
        self._cached_batch = batch
        self._cached_weights = weights
        self._num_events = len(table)

        return batch, indices

    # _apply_feature_system 已废弃，现在使用 FeatureGraph.build_batch() 替代

    def __iter__(self):
        """迭代器：返回数据批次。

        Yields:
            Dict[str, Any]: 数据批次（包含特征张量、标签、观察者）
        """
        # 加载并预处理
        batch, indices = self._load_and_preprocess()

        if len(indices) == 0:
            return

        # batch 已经是 torch.Tensor 字典（从 feature_graph.build_batch 返回）
        # 需要按索引提取单个样本
        for idx in indices:
            sample = {}
            for key, value in batch.items():
                try:
                    if isinstance(value, torch.Tensor):
                        # 提取单个样本
                        sample[key] = value[idx]
                    elif isinstance(value, ak.Array):
                        # 观察者变量（ak.Array）
                        sample[key] = value[idx]
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
        # 如果缓存已初始化，使用缓存的事件数
        if self._num_events > 0:
            return self._num_events

        # 否则尝试从数据源获取
        num_events = self.data_source.get_num_events()
        if num_events is not None:
            return num_events
        else:
            # 如果无法确定，返回一个估计值
            return 0

    def get_sample(self, index: int) -> dict[str, Any]:
        """获取单个样本（用于调试）。

        Args:
            index: 样本索引

        Returns:
            Dict[str, Any]: 样本数据
        """
        batch, indices = self._load_and_preprocess()

        if index >= len(indices):
            raise IndexError(f"Index {index} out of range")

        idx = indices[index]
        sample = {}
        for key, value in batch.items():
            try:
                if isinstance(value, torch.Tensor):
                    sample[key] = value[idx]
                elif isinstance(value, ak.Array):
                    sample[key] = value[idx]
                else:
                    sample[key] = value
            except (IndexError, KeyError):
                continue

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

    def _format_transformer_input(self, sample: dict[str, Any]) -> dict[str, torch.Tensor]:
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
                if isinstance(label_data, torch.Tensor):
                    # Already a tensor, just use it
                    output[label_name] = label_data
                elif isinstance(label_data, ak.Array):
                    label_data = ak.to_numpy(label_data)
                    output[label_name] = torch.from_numpy(label_data.astype(np.int64))
                elif isinstance(label_data, np.ndarray):
                    output[label_name] = torch.from_numpy(label_data.astype(np.int64))
                else:
                    # Scalar or other type
                    output[label_name] = torch.tensor(label_data, dtype=torch.int64)

        return output

    def __iter__(self):
        """迭代器：返回 Transformer 格式的数据批次。

        Yields:
            Dict[str, torch.Tensor]: Transformer 格式的数据
        """
        for sample in super().__iter__():
            yield self._format_transformer_input(sample)
