"""
数据配置模块

完全借鉴 weaver-core 的 DataConfig 类，提供：
- YAML 配置驱动的特征定义
- 变量表达式系统
- 输入/标签/权重配置
- 预处理参数管理
"""
import copy

import numpy as np
import yaml

from .logger import _logger
from .tools import _get_variable_names


def _as_list(x):
    """将输入转换为列表。

    Args:
        x: 输入值。

    Returns:
        list: 列表形式的值。
    """
    if x is None:
        return None
    elif isinstance(x, (list, tuple)):
        return x
    else:
        return [x]


def _md5(fname):
    """计算文件的 MD5 哈希值。

    Args:
        fname (str): 文件路径。

    Returns:
        str: MD5 哈希值的十六进制字符串。
    """
    import hashlib

    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


class DataConfig(object):
    """
    数据配置类，用于存储数据加载器的配置。

    完全借鉴 weaver-core 的设计，支持：
    - Config-driven 特征定义
    - expr 表达式生成新变量
    - 自动变量计算、标准化、裁剪、padding
    - 支持 sequence / mask / transformer 输入
    """

    def __init__(self, print_info=True, **kwargs):
        opts = {
            "treename": None,
            "branch_magic": None,
            "file_magic": None,
            "selection": None,
            "test_time_selection": None,
            "preprocess": {"method": "manual", "data_fraction": 0.1, "params": None},
            "new_variables": {},
            "inputs": {},
            "labels": {},
            "observers": [],
            "monitor_variables": [],
            "weights": None,
        }

        for k, v in kwargs.items():
            if v is not None:
                if isinstance(opts[k], dict):
                    opts[k].update(v)
                else:
                    opts[k] = v

        self.options = opts
        if print_info:
            _logger.debug(opts)

        self.train_load_branches = set()
        self.train_aux_branches = set()
        self.test_load_branches = set()
        self.test_aux_branches = set()

        self.selection = opts["selection"]
        self.test_time_selection = opts["test_time_selection"] if opts["test_time_selection"] else self.selection
        self.var_funcs = copy.deepcopy(opts["new_variables"])

        # 预处理配置
        self.preprocess = opts["preprocess"]
        self._auto_standardization = opts["preprocess"]["method"].lower().startswith("auto")
        self._missing_standardization_info = False
        self.preprocess_params = opts["preprocess"]["params"] if opts["preprocess"]["params"] is not None else {}

        # 输入配置
        self.input_names = tuple(opts["inputs"].keys())
        self.input_dicts = {k: [] for k in self.input_names}
        self.input_shapes = {}

        for k, o in opts["inputs"].items():
            self.input_shapes[k] = (-1, len(o["vars"]), o["length"])
            for v in o["vars"]:
                v = _as_list(v)
                self.input_dicts[k].append(v[0])

                if opts["preprocess"]["params"] is None:

                    def _get(idx, default):
                        try:
                            return v[idx]
                        except IndexError:
                            return default

                    params = {
                        "length": o["length"],
                        "pad_mode": o.get("pad_mode", "constant").lower(),
                        "center": _get(1, "auto" if self._auto_standardization else None),
                        "scale": _get(2, 1),
                        "min": _get(3, -5),
                        "max": _get(4, 5),
                        "pad_value": _get(5, 0),
                    }

                    if v[0] in self.preprocess_params and params != self.preprocess_params[v[0]]:
                        raise RuntimeError(
                            "变量 %s 的信息不兼容，已有: \n  %s\n现在得到:\n  %s" % (v[0], str(self.preprocess_params[v[0]]), str(params))
                        )

                    if k.endswith("_mask") and params["pad_mode"] != "constant":
                        raise RuntimeError("掩码输入 `%s` 的 `pad_mode` 必须设置为 `constant`" % k)

                    if params["center"] == "auto":
                        self._missing_standardization_info = True

                    self.preprocess_params[v[0]] = params

        # 标签配置
        self.label_type = opts["labels"]["type"]
        self.label_value = opts["labels"]["value"]
        if self.label_type == "simple":
            assert isinstance(self.label_value, list)
            self.label_names = ("_label_",)
            label_exprs = ["ak.to_numpy(%s)" % k for k in self.label_value]
            self.register("_label_", "np.argmax(np.stack([%s], axis=1), axis=1)" % (",".join(label_exprs)))
            self.register("_labelcheck_", "np.sum(np.stack([%s], axis=1), axis=1)", "train")
        else:
            self.label_names = tuple(self.label_value.keys())
            self.register(self.label_value)

        # 权重配置
        self.basewgt_name = "_basewgt_"
        self.weight_name = None
        if opts["weights"] is not None:
            self.weight_name = "_weight_"
            self.use_precomputed_weights = opts["weights"]["use_precomputed_weights"]
            if self.use_precomputed_weights:
                self.register(self.weight_name, "*".join(opts["weights"]["weight_branches"]), "train")
            else:
                self.reweight_method = opts["weights"]["reweight_method"]
                self.reweight_basewgt = opts["weights"].get("reweight_basewgt", None)
                if self.reweight_basewgt:
                    self.register(self.basewgt_name, self.reweight_basewgt, "train")
                self.reweight_branches = tuple(opts["weights"]["reweight_vars"].keys())
                self.reweight_bins = tuple(opts["weights"]["reweight_vars"].values())
                self.reweight_classes = tuple(opts["weights"]["reweight_classes"])
                self.register(self.reweight_branches + self.reweight_classes, to="train")
                self.class_weights = opts["weights"].get("class_weights", None)
                if self.class_weights is None:
                    self.class_weights = np.ones(len(self.reweight_classes))
                self.reweight_threshold = opts["weights"].get("reweight_threshold", 10)
                self.reweight_discard_under_overflow = opts["weights"].get("reweight_discard_under_overflow", True)
                self.reweight_hists = opts["weights"].get("reweight_hists", None)
                if self.reweight_hists is not None:
                    for k, v in self.reweight_hists.items():
                        self.reweight_hists[k] = np.array(v, dtype="float32")

        # 观察者变量
        self.observer_names = tuple(opts["observers"])
        self.monitor_variables = tuple(opts["monitor_variables"])
        if self.observer_names and self.monitor_variables:
            raise RuntimeError("不能同时设置 `observers` 和 `monitor_variables`。")
        self.z_variables = self.observer_names if len(self.observer_names) > 0 else self.monitor_variables

        # 移除自映射
        for k, v in list(self.var_funcs.items()):
            if k == v:
                del self.var_funcs[k]

        if print_info:

            def _log(msg, *args, **kwargs):
                _logger.info(msg, *args, **kwargs)

            _log("预处理配置: %s", str(self.preprocess))
            _log("选择条件: %s", str(self.selection))
            _log("测试时选择条件: %s", str(self.test_time_selection))
            _log("变量函数:\n - %s", "\n - ".join(str(it) for it in self.var_funcs.items()))
            _log("输入名称: %s", str(self.input_names))
            _log("输入字典:\n - %s", "\n - ".join(str(it) for it in self.input_dicts.items()))
            _log("输入形状:\n - %s", "\n - ".join(str(it) for it in self.input_shapes.items()))
            _log("预处理参数:\n - %s", "\n - ".join(str(it) for it in self.preprocess_params.items()))
            _log("标签名称: %s", str(self.label_names))
            _log("观察者名称: %s", str(self.observer_names))
            _log("监控变量: %s", str(self.monitor_variables))
            if opts["weights"] is not None:
                if self.use_precomputed_weights:
                    _log("权重: %s" % self.var_funcs[self.weight_name])
                else:
                    for k in [
                        "reweight_method",
                        "reweight_basewgt",
                        "reweight_branches",
                        "reweight_bins",
                        "reweight_classes",
                        "class_weights",
                        "reweight_threshold",
                        "reweight_discard_under_overflow",
                    ]:
                        _log("%s: %s" % (k, getattr(self, k)))

        # 注册依赖
        if self.selection:
            self.register(_get_variable_names(self.selection), to="train")
        if self.test_time_selection:
            self.register(_get_variable_names(self.test_time_selection), to="test")
        for names in self.input_dicts.values():
            self.register(names)
        self.register(self.observer_names, to="test")
        self.register(self.monitor_variables)

        # 解析依赖关系
        func_vars = set(self.var_funcs.keys())
        for load_branches, aux_branches in (self.train_load_branches, self.train_aux_branches), (
            self.test_load_branches,
            self.test_aux_branches,
        ):
            while load_branches & func_vars:
                for k in load_branches & func_vars:
                    aux_branches.add(k)
                    load_branches.remove(k)
                    # 跳过包含占位符的表达式（如 _labelcheck_）
                    expr = self.var_funcs[k]
                    if "%s" in str(expr):
                        continue
                    try:
                        load_branches.update(_get_variable_names(expr))
                    except (SyntaxError, ValueError):
                        # 如果表达式无法解析，跳过
                        continue

        if print_info:
            _logger.debug("训练加载分支:\n  %s", ", ".join(sorted(self.train_load_branches)))
            _logger.debug("训练辅助分支:\n  %s", ", ".join(sorted(self.train_aux_branches)))
            _logger.debug("测试加载分支:\n  %s", ", ".join(sorted(self.test_load_branches)))
            _logger.debug("测试辅助分支:\n  %s", ", ".join(sorted(self.test_aux_branches)))

    def __getattr__(self, name):
        return self.options[name]

    def register(self, name, expr=None, to="both"):
        """注册变量或表达式。

        Args:
            name: 变量名（可以是字符串、列表、字典）。
            expr (str, optional): 表达式。默认为 None。
            to (str): 注册到哪个集合（'train', 'test', 'both'）。默认为 'both'。
        """
        assert to in ("train", "test", "both")
        if isinstance(name, dict):
            for k, v in name.items():
                self.register(k, v, to)
        elif isinstance(name, (list, tuple)):
            for k in name:
                self.register(k, None, to)
        else:
            if to in ("train", "both"):
                self.train_load_branches.add(name)
            if to in ("test", "both"):
                self.test_load_branches.add(name)
            if expr:
                self.var_funcs[name] = expr
                if to in ("train", "both"):
                    self.train_aux_branches.add(name)
                if to in ("test", "both"):
                    self.test_aux_branches.add(name)

    def dump(self, fp):
        """将配置保存到 YAML 文件。

        Args:
            fp (str): 文件路径。
        """
        with open(fp, "w") as f:
            yaml.safe_dump(self.options, f, sort_keys=False, allow_unicode=True)

    @classmethod
    def load(cls, fp, load_observers=True, load_reweight_info=True, extra_selection=None, extra_test_selection=None):
        """从 YAML 文件加载配置。

        Args:
            fp (str): 文件路径。
            load_observers (bool): 是否加载观察者。默认为 True。
            load_reweight_info (bool): 是否加载重加权信息。默认为 True。
            extra_selection (str, optional): 额外的选择条件。默认为 None。
            extra_test_selection (str, optional): 额外的测试时选择条件。默认为 None。

        Returns:
            DataConfig: 数据配置对象。
        """
        with open(fp) as f:
            _opts = yaml.safe_load(f)
            options = copy.deepcopy(_opts)
        if not load_observers:
            options["observers"] = None
        if not load_reweight_info:
            options["weights"] = None
        if extra_selection:
            options["selection"] = "(%s) & (%s)" % (_opts["selection"], extra_selection)
        if extra_test_selection:
            if "test_time_selection" not in options or options["test_time_selection"] is None:
                options["test_time_selection"] = "(%s) & (%s)" % (_opts["selection"], extra_test_selection)
            else:
                options["test_time_selection"] = "(%s) & (%s)" % (_opts["test_time_selection"], extra_test_selection)
        return cls(**options)

    def copy(self):
        """复制配置。

        Returns:
            DataConfig: 配置副本。
        """
        return self.__class__(print_info=False, **copy.deepcopy(self.options))

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy()

    def export_json(self, fp):
        """导出为 JSON 格式（用于推理）。

        Args:
            fp (str): 文件路径。
        """
        import json

        j = {"output_names": self.label_value, "input_names": self.input_names}
        for k, v in self.input_dicts.items():
            j[k] = {"var_names": v, "var_infos": {}}
            for var_name in v:
                j[k]["var_length"] = self.preprocess_params[var_name]["length"]
                info = self.preprocess_params[var_name]
                j[k]["var_infos"][var_name] = {
                    "median": 0 if info["center"] is None else info["center"],
                    "norm_factor": info["scale"],
                    "replace_inf_value": 0,
                    "lower_bound": -1e32 if info["center"] is None else info["min"],
                    "upper_bound": 1e32 if info["center"] is None else info["max"],
                    "pad": info["pad_value"],
                }
        with open(fp, "w") as f:
            json.dump(j, f, indent=2)
