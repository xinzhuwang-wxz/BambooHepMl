"""
数据配置模块

重构后的 DataConfig 类，只负责数据源配置：
- 数据源路径和树名
- 选择条件（selection）
- 标签定义（labels）
- 权重配置（weights）
- 观察者变量（observers）

注意：特征定义已迁移到 FeatureGraph，不再在 DataConfig 中定义。
"""

import copy

import numpy as np
import yaml

from .logger import _logger
from .tools import _get_variable_names


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


class DataConfig:
    """
    数据配置类，用于存储数据加载器的配置。

    重构后只负责数据源配置，不再定义特征：
    - 数据源：treename, branch_magic, file_magic
    - 选择条件：selection, test_time_selection
    - 标签：labels（用于训练目标）
    - 权重：weights（用于样本重加权）
    - 观察者：observers, monitor_variables

    特征定义现在由 FeatureGraph 管理（通过 features.yaml）。
    """

    def __init__(self, print_info=True, **kwargs):
        opts = {
            "treename": None,
            "branch_magic": None,
            "file_magic": None,
            "selection": None,
            "test_time_selection": None,
            "labels": {},
            "observers": [],
            "monitor_variables": [],
            "weights": None,
        }

        # 检查是否传入了废弃的参数
        deprecated_params = ["new_variables", "inputs", "preprocess"]
        for param in deprecated_params:
            if param in kwargs and kwargs[param] is not None:
                _logger.warning(
                    f"Parameter '{param}' is deprecated. "
                    f"Feature definitions should now be in features.yaml via FeatureGraph. "
                    f"Ignoring '{param}' parameter."
                )
                kwargs.pop(param)

        for k, v in kwargs.items():
            if v is not None:
                if isinstance(opts.get(k), dict):
                    opts[k].update(v)
                else:
                    opts[k] = v

        self.options = opts
        if print_info:
            _logger.debug(opts)

        # 分支加载集合（用于确定需要从 ROOT/Parquet/HDF5 加载哪些原始分支）
        self.train_load_branches = set()
        self.train_aux_branches = set()
        self.test_load_branches = set()
        self.test_aux_branches = set()

        # 选择条件
        self.selection = opts["selection"]
        self.test_time_selection = opts["test_time_selection"] if opts["test_time_selection"] else self.selection

        # 变量函数（仅用于 labels/weights/selection 的表达式，不是特征）
        self.var_funcs = {}

        # 标签配置
        self.label_type = opts["labels"].get("type") if opts["labels"] else None
        self.label_value = opts["labels"].get("value") if opts["labels"] else None
        self.label_classes = opts["labels"].get("classes") if opts["labels"] else None

        # class_names: ordered list of class names (e.g. ["pi_3GeV", "pi_5GeV", "pi_7GeV"])
        # class_files: list of glob patterns per class, aligned with class_names
        self.class_names = None
        self.class_files = None

        if self.label_type and self.label_classes:
            # New classes-based label: each file = one class, _label_ is int index
            assert self.label_type == "simple", "labels.classes is only supported for labels.type=simple (classification)"
            assert isinstance(self.label_classes, list), "labels.classes must be a list"
            self.class_names = [c["name"] for c in self.label_classes]
            self.class_files = [c["files"] for c in self.label_classes]
            self.label_names = ("_label_",)
            self.label_value = None  # not used in classes mode
            # No var_funcs needed — _label_ is injected directly by DataSource
        elif self.label_type and self.label_value:
            if self.label_type == "simple":
                assert isinstance(self.label_value, list)
                self.label_names = ("_label_",)
                # 构建标签表达式（用于从原始分支计算标签）
                label_exprs = ["ak.to_numpy(%s)" % k for k in self.label_value]
                self.register(
                    "_label_",
                    "np.argmax(np.stack([%s], axis=1), axis=1)" % (",".join(label_exprs)),
                )
                self.register("_labelcheck_", "np.sum(np.stack([%s], axis=1), axis=1)", "train")
            else:
                self.label_names = tuple(self.label_value.keys())
                self.register(self.label_value)
        else:
            self.label_names = tuple()

        # 权重配置
        self.basewgt_name = "_basewgt_"
        self.weight_name = None
        if opts["weights"] is not None:
            self.weight_name = "_weight_"
            self.use_precomputed_weights = opts["weights"].get("use_precomputed_weights", False)
            if self.use_precomputed_weights:
                self.register(
                    self.weight_name,
                    "*".join(opts["weights"]["weight_branches"]),
                    "train",
                )
            else:
                self.reweight_method = opts["weights"].get("reweight_method")
                self.reweight_basewgt = opts["weights"].get("reweight_basewgt", None)
                if self.reweight_basewgt:
                    self.register(self.basewgt_name, self.reweight_basewgt, "train")
                self.reweight_branches = tuple(opts["weights"].get("reweight_vars", {}).keys())
                self.reweight_bins = tuple(opts["weights"].get("reweight_vars", {}).values())
                self.reweight_classes = tuple(opts["weights"].get("reweight_classes", []))
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

            _log("选择条件: %s", str(self.selection))
            _log("测试时选择条件: %s", str(self.test_time_selection))
            _log("标签名称: %s", str(self.label_names))
            _log("观察者名称: %s", str(self.observer_names))
            _log("监控变量: %s", str(self.monitor_variables))
            if opts["weights"] is not None:
                if self.use_precomputed_weights:
                    _log("权重: %s" % self.var_funcs.get(self.weight_name, "N/A"))
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
                        if hasattr(self, k):
                            _log(f"{k}: {getattr(self, k)}")

        # 注册依赖（用于确定需要加载哪些原始分支）
        if self.selection:
            self.register(_get_variable_names(self.selection), to="train")
        if self.test_time_selection:
            self.register(_get_variable_names(self.test_time_selection), to="test")
        self.register(self.observer_names, to="test")
        self.register(self.monitor_variables)

        # 解析依赖关系（用于确定 aux_branches，即需要计算的变量）
        func_vars = set(self.var_funcs.keys())
        for load_branches, aux_branches in (
            (self.train_load_branches, self.train_aux_branches),
            (
                self.test_load_branches,
                self.test_aux_branches,
            ),
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
        return self.options.get(name)

    def register(self, name, expr=None, to="both"):
        """注册变量或表达式（仅用于 labels/weights/selection，不用于特征）。

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
    def load(
        cls,
        fp,
        load_observers=True,
        load_reweight_info=True,
        extra_selection=None,
        extra_test_selection=None,
    ):
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

        # 移除废弃的参数
        for deprecated in ["new_variables", "inputs", "preprocess"]:
            if deprecated in options:
                _logger.warning(f"Deprecated parameter '{deprecated}' found in config file. Ignoring it.")
                options.pop(deprecated)

        if not load_observers:
            options["observers"] = None
        if not load_reweight_info:
            options["weights"] = None
        if extra_selection:
            if options.get("selection"):
                options["selection"] = "({}) & ({})".format(_opts["selection"], extra_selection)
            else:
                options["selection"] = extra_selection
        if extra_test_selection:
            if "test_time_selection" not in options or options["test_time_selection"] is None:
                if options.get("selection"):
                    options["test_time_selection"] = "({}) & ({})".format(_opts.get("selection", ""), extra_test_selection)
                else:
                    options["test_time_selection"] = extra_test_selection
            else:
                options["test_time_selection"] = "({}) & ({})".format(options["test_time_selection"], extra_test_selection)
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
