"""
预处理模块

提供：
- 自动标准化（AutoStandardizer）
- 权重生成（WeightMaker）
- 选择条件应用
- 新变量构建
"""

import copy
import glob
import time

import awkward as ak
import numpy as np

from .fileio import read_files
from .logger import _logger, warn_n_times
from .tools import _eval_expr, _get_variable_names


def _apply_selection(table, selection, funcs=None):
    """应用选择条件。

    Args:
        table: awkward 数组。
        selection (str): 选择条件表达式。
        funcs (dict, optional): 新变量函数字典 {var_name: expr}。默认为 None。

    Returns:
        ak.Array: 选择后的数组。
    """
    if selection is None:
        return table
    if funcs:
        new_vars = {k: funcs[k] for k in _get_variable_names(selection) if k not in table.fields and k in funcs}
        _build_new_variables(table, new_vars)
    selected = ak.values_astype(_eval_expr(selection, table), "bool")
    return table[selected]


def _build_new_variables(table, funcs):
    """构建新变量。

    Args:
        table: awkward 数组。
        funcs (dict): 变量函数字典 {var_name: expr}。

    Returns:
        ak.Array: 更新后的数组。
    """
    if funcs is None:
        return table

    # 过滤掉已经存在的变量
    funcs = {k: v for k, v in funcs.items() if k not in table.fields}
    if not funcs:
        return table

    # 构建依赖图并确定构造顺序
    from bamboohepml.data.tools import _get_variable_names

    # 构建依赖关系：var_name -> [依赖的变量名列表]
    dependencies = {}
    for var_name, expr in funcs.items():
        # 跳过包含占位符的表达式（如 _labelcheck_）
        if "%s" in str(expr):
            continue
        try:
            deps = _get_variable_names(expr)
            # 只保留在 funcs 中的依赖（即新变量之间的依赖）
            dependencies[var_name] = [d for d in deps if d in funcs]
        except (SyntaxError, ValueError):
            # 如果表达式无法解析，假设无依赖
            dependencies[var_name] = []

    # 拓扑排序：确定构造顺序
    # 使用 Kahn 算法
    in_degree = {var_name: len(deps) for var_name, deps in dependencies.items()}
    queue = [var_name for var_name, degree in in_degree.items() if degree == 0]
    execution_order = []

    while queue:
        var_name = queue.pop(0)
        execution_order.append(var_name)

        # 减少依赖此变量的变量的入度
        for other_var, deps in dependencies.items():
            if var_name in deps:
                in_degree[other_var] -= 1
                if in_degree[other_var] == 0:
                    queue.append(other_var)

    # 添加剩余的变量（无依赖关系的）
    remaining = set(funcs.keys()) - set(execution_order)
    execution_order.extend(remaining)

    # 按顺序构造变量
    for var_name in execution_order:
        if var_name not in funcs:
            continue
        if var_name in table.fields:
            continue
        expr = funcs[var_name]
        # 跳过包含占位符的表达式
        if "%s" in str(expr):
            continue
        try:
            table[var_name] = _eval_expr(expr, table)
        except Exception as e:
            from bamboohepml.data.logger import _logger

            _logger.warning(f"Failed to build variable {var_name}: {e}")
            raise

    return table


def _build_weights(table, data_config, reweight_hists=None):
    """构建权重。

    Args:
        table: awkward 数组。
        data_config: DataConfig 对象。
        reweight_hists (dict, optional): 重加权直方图。默认为 None。

    Returns:
        np.ndarray: 权重数组。
    """
    if data_config.weight_name is None:
        raise RuntimeError("构建权重时出错：`weight_name` 为 None！")
    if data_config.use_precomputed_weights:
        return ak.to_numpy(table[data_config.weight_name])
    else:
        x_var, y_var = data_config.reweight_branches
        x_bins, y_bins = data_config.reweight_bins
        rwgt_sel = None
        if data_config.reweight_discard_under_overflow:
            rwgt_sel = (table[x_var] >= min(x_bins)) & (table[x_var] <= max(x_bins)) & (table[y_var] >= min(y_bins)) & (table[y_var] <= max(y_bins))
        wgt = np.zeros(len(table), dtype="float32")
        sum_evts = 0
        if reweight_hists is None:
            reweight_hists = data_config.reweight_hists
        for label, hist in reweight_hists.items():
            pos = table[label] == 1
            if rwgt_sel is not None:
                pos = pos & rwgt_sel
            rwgt_x_vals = ak.to_numpy(table[x_var][pos])
            rwgt_y_vals = ak.to_numpy(table[y_var][pos])
            x_indices = np.clip(np.digitize(rwgt_x_vals, x_bins) - 1, a_min=0, a_max=len(x_bins) - 2)
            y_indices = np.clip(np.digitize(rwgt_y_vals, y_bins) - 1, a_min=0, a_max=len(y_bins) - 2)
            wgt[pos] = hist[x_indices, y_indices]
            sum_evts += np.sum(pos)
        if sum_evts != len(table):
            warn_n_times(
                "并非所有选择的事件都用于重加权。"
                "请检查 `selection` 和 `reweight_classes` 定义之间的一致性，或与 `reweight_vars` 分箱的一致性"
                "（默认情况下丢弃下溢和上溢箱，除非在 `weights` 部分将 `reweight_discard_under_overflow` 设置为 `False`）。",
            )
        if data_config.reweight_basewgt:
            wgt *= ak.to_numpy(table[data_config.basewgt_name])
        return wgt


class AutoStandardizer(object):
    """自动标准化器。

    用于计算变量标准化信息（中心值和缩放因子）。
    """

    def __init__(self, filelist, data_config):
        """初始化自动标准化器。

        Args:
            filelist: 文件列表（可以是列表、字典或 glob 模式）。
            data_config: DataConfig 对象。
        """
        if isinstance(filelist, dict):
            filelist = sum(filelist.values(), [])
        self._filelist = filelist if isinstance(filelist, (list, tuple)) else glob.glob(filelist)
        self._data_config = data_config.copy()
        self.load_range = (0, data_config.preprocess.get("data_fraction", 0.1))

    def read_file(self, filelist):
        """读取文件并应用选择条件。

        Args:
            filelist: 文件列表。

        Returns:
            ak.Array: 处理后的数据。
        """
        keep_branches = set()
        aux_branches = set()
        load_branches = set()

        for k, params in self._data_config.preprocess_params.items():
            if params["center"] == "auto":
                keep_branches.add(k)
                load_branches.add(k)

        if self._data_config.selection:
            load_branches.update(_get_variable_names(self._data_config.selection))

        func_vars = set(self._data_config.var_funcs.keys())
        while load_branches & func_vars:
            for k in load_branches & func_vars:
                aux_branches.add(k)
                load_branches.remove(k)
                load_branches.update(_get_variable_names(self._data_config.var_funcs[k]))

        _logger.debug("[AutoStandardizer] 保留分支:\n  %s", ",".join(keep_branches))
        _logger.debug("[AutoStandardizer] 辅助分支:\n  %s", ",".join(aux_branches))
        _logger.debug("[AutoStandardizer] 加载分支:\n  %s", ",".join(load_branches))

        if not load_branches:
            _logger.warning("[AutoStandardizer] 没有需要加载的分支！")
            return ak.Array({})

        table = read_files(
            filelist,
            load_branches,
            self.load_range,
            show_progressbar=True,
            treename=self._data_config.treename,
            branch_magic=self._data_config.branch_magic,
            file_magic=self._data_config.file_magic,
        )

        if len(table) == 0:
            _logger.warning("[AutoStandardizer] 读取的数据为空！load_range=%s", self.load_range)
            return table

        table = _apply_selection(table, self._data_config.selection, funcs=self._data_config.var_funcs)
        table = _build_new_variables(table, {k: v for k, v in self._data_config.var_funcs.items() if k in aux_branches})

        if keep_branches:
            table = table[list(keep_branches)]
        else:
            _logger.warning("[AutoStandardizer] keep_branches 为空，返回完整 table")

        return table

    def make_preprocess_params(self, table):
        """生成预处理参数。

        Args:
            table: 数据表。

        Returns:
            dict: 预处理参数字典。
        """
        _logger.info("使用 %d 个事件计算标准化信息", len(table))
        preprocess_params = copy.deepcopy(self._data_config.preprocess_params)

        for k, params in self._data_config.preprocess_params.items():
            if params["center"] == "auto":
                if k.endswith("_mask"):
                    params["center"] = None
                else:
                    a = ak.to_numpy(ak.flatten(table[k], axis=None))
                    if np.any(np.isnan(a)):
                        _logger.warning("[AutoStandardizer] 在 `%s` 中发现 NaN，将转换为 0。", k)
                        time.sleep(10)
                        a = np.nan_to_num(a)
                    low, center, high = np.percentile(a, [16, 50, 84])
                    scale = max(high - center, center - low)
                    scale = 1 if scale == 0 else 1.0 / scale
                    params["center"] = float(center)
                    params["scale"] = float(scale)
                preprocess_params[k] = params
                _logger.info("[AutoStandardizer] %s low=%s, center=%s, high=%s, scale=%s", k, low, center, high, scale)

        return preprocess_params

    def produce(self, output=None):
        """生成预处理参数并保存。

        Args:
            output (str, optional): 输出文件路径。默认为 None。

        Returns:
            DataConfig: 更新后的数据配置。
        """
        table = self.read_file(self._filelist)
        preprocess_params = self.make_preprocess_params(table)
        self._data_config.preprocess_params = preprocess_params
        self._data_config.options["preprocess"]["params"] = preprocess_params
        if output:
            _logger.info("将自动生成的预处理信息写入 YAML 文件: %s" % output)
            self._data_config.dump(output)
        return self._data_config


class WeightMaker(object):
    """权重生成器。

    用于生成重加权信息。
    """

    def __init__(self, filelist, data_config):
        """初始化权重生成器。

        Args:
            filelist: 文件列表。
            data_config: DataConfig 对象。
        """
        if isinstance(filelist, dict):
            filelist = sum(filelist.values(), [])
        self._filelist = filelist if isinstance(filelist, (list, tuple)) else glob.glob(filelist)
        self._data_config = data_config.copy()

    def read_file(self, filelist):
        """读取文件。

        Args:
            filelist: 文件列表。

        Returns:
            ak.Array: 处理后的数据。
        """
        keep_branches = set(self._data_config.reweight_branches + self._data_config.reweight_classes)
        if self._data_config.reweight_basewgt:
            keep_branches.add(self._data_config.basewgt_name)
        aux_branches = set()
        load_branches = keep_branches.copy()
        if self._data_config.selection:
            load_branches.update(_get_variable_names(self._data_config.selection))

        func_vars = set(self._data_config.var_funcs.keys())
        while load_branches & func_vars:
            for k in load_branches & func_vars:
                aux_branches.add(k)
                load_branches.remove(k)
                load_branches.update(_get_variable_names(self._data_config.var_funcs[k]))

        _logger.debug("[WeightMaker] 保留分支:\n  %s", ",".join(keep_branches))
        _logger.debug("[WeightMaker] 辅助分支:\n  %s", ",".join(aux_branches))
        _logger.debug("[WeightMaker] 加载分支:\n  %s", ",".join(load_branches))

        table = read_files(
            filelist,
            load_branches,
            show_progressbar=True,
            treename=self._data_config.treename,
            branch_magic=self._data_config.branch_magic,
            file_magic=self._data_config.file_magic,
        )
        table = _apply_selection(table, self._data_config.selection, funcs=self._data_config.var_funcs)
        table = _build_new_variables(table, {k: v for k, v in self._data_config.var_funcs.items() if k in aux_branches})
        table = table[keep_branches]
        return table

    def make_weights(self, table):
        """生成权重直方图。

        Args:
            table: 数据表。

        Returns:
            dict: 权重直方图字典。
        """
        x_var, y_var = self._data_config.reweight_branches
        x_bins, y_bins = self._data_config.reweight_bins
        if not self._data_config.reweight_discard_under_overflow:
            x_min, x_max = min(x_bins), max(x_bins)
            y_min, y_max = min(y_bins), max(y_bins)
            _logger.info(f"将 `{x_var}` 裁剪到 [{x_min}, {x_max}] 以计算重加权形状。")
            _logger.info(f"将 `{y_var}` 裁剪到 [{y_min}, {y_max}] 以计算重加权形状。")
            table[x_var] = np.clip(table[x_var], min(x_bins), max(x_bins))
            table[y_var] = np.clip(table[y_var], min(y_bins), max(y_bins))

        _logger.info("使用 %d 个事件生成权重", len(table))

        sum_evts = 0
        max_weight = 0.9
        raw_hists = {}
        class_events = {}
        result = {}

        for label in self._data_config.reweight_classes:
            pos = table[label] == 1
            x = ak.to_numpy(table[x_var][pos])
            y = ak.to_numpy(table[y_var][pos])
            hist, _, _ = np.histogram2d(x, y, bins=self._data_config.reweight_bins)
            _logger.info("%s (未加权):\n %s", label, str(hist.astype("int64")))
            sum_evts += hist.sum()
            if self._data_config.reweight_basewgt:
                w = ak.to_numpy(table[self._data_config.basewgt_name][pos])
                hist, _, _ = np.histogram2d(x, y, weights=w, bins=self._data_config.reweight_bins)
                _logger.info("%s (加权):\n %s", label, str(hist.astype("float32")))
            raw_hists[label] = hist.astype("float32")
            result[label] = hist.astype("float32")

        if sum_evts != len(table):
            _logger.warning(
                "只有 %d（共 %d）个事件实际用于重加权。"
                "请检查 `selection` 和 `reweight_classes` 定义之间的一致性，或与 `reweight_vars` 分箱的一致性"
                "（默认情况下丢弃下溢和上溢箱，除非在 `weights` 部分将 `reweight_discard_under_overflow` 设置为 `False`）。",
                sum_evts,
                len(table),
            )
            time.sleep(10)

        if self._data_config.reweight_method == "flat":
            for label, classwgt in zip(self._data_config.reweight_classes, self._data_config.class_weights):
                hist = result[label]
                threshold_ = np.median(hist[hist > 0]) * 0.01
                nonzero_vals = hist[hist > threshold_]
                min_val, med_val = np.min(nonzero_vals), np.median(hist)
                ref_val = np.percentile(nonzero_vals, self._data_config.reweight_threshold)
                _logger.debug("label:%s, median=%f, min=%f, ref=%f, ref/min=%f" % (label, med_val, min_val, ref_val, ref_val / min_val))
                wgt = np.clip(np.nan_to_num(ref_val / hist, posinf=0), 0, 1)
                result[label] = wgt
                class_events[label] = np.sum(raw_hists[label] * wgt) / classwgt
        elif self._data_config.reweight_method == "ref":
            hist_ref = raw_hists[self._data_config.reweight_classes[0]]
            for label, classwgt in zip(self._data_config.reweight_classes, self._data_config.class_weights):
                ratio = np.nan_to_num(hist_ref / result[label], posinf=0)
                upper = np.percentile(ratio[ratio > 0], 100 - self._data_config.reweight_threshold)
                wgt = np.clip(ratio / upper, 0, 1)
                result[label] = wgt
                class_events[label] = np.sum(raw_hists[label] * wgt) / classwgt

        min_nevt = min(class_events.values()) * max_weight
        for label in self._data_config.reweight_classes:
            class_wgt = float(min_nevt) / class_events[label]
            result[label] *= class_wgt

        if self._data_config.reweight_basewgt:
            wgts = _build_weights(table, self._data_config, reweight_hists=result)
            _logger.info("样本权重百分位数: %s", str(np.percentile(wgts, np.arange(101))))
            wgt_ref = np.percentile(wgts, 100 - self._data_config.reweight_threshold)
            _logger.info("设置整体重加权缩放因子（%d 阈值）为 %s（最大 %s）" % (100 - self._data_config.reweight_threshold, wgt_ref, np.max(wgts)))
            for label in self._data_config.reweight_classes:
                result[label] /= wgt_ref

        _logger.info("权重:")
        for label in self._data_config.reweight_classes:
            _logger.info("%s:\n %s", label, str(result[label]))

        _logger.info("原始直方图 * 权重:")
        for label in self._data_config.reweight_classes:
            _logger.info("%s:\n %s", label, str((raw_hists[label] * result[label]).astype("int32")))

        return result

    def produce(self, output=None):
        """生成权重并保存。

        Args:
            output (str, optional): 输出文件路径。默认为 None。

        Returns:
            DataConfig: 更新后的数据配置。
        """
        table = self.read_file(self._filelist)
        wgts = self.make_weights(table)
        self._data_config.reweight_hists = wgts
        self._data_config.options["weights"]["reweight_hists"] = {k: v.tolist() for k, v in wgts.items()}
        if output:
            _logger.info("将重加权信息写入 YAML 文件: %s" % output)
            self._data_config.dump(output)
        return self._data_config
