"""
Pipeline Orchestrator

统一入口，协调整个 ML pipeline。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..config import logger
from ..data import DataConfig, DataSourceFactory, HEPDataset
from ..data.features import ExpressionEngine, FeatureGraph
from ..models import get_model
from .state import PipelineState


class PipelineOrchestrator:
    """
    Pipeline 编排器，负责协调整个 ML pipeline 的生命周期。

    作为框架的统一入口点，PipelineOrchestrator 负责：
    - 加载和解析 pipeline.yaml 配置文件
    - 初始化数据源、特征图、模型等组件
    - 验证配置一致性和组件兼容性
    - 管理 PipelineState（用于配置验证和持久化）

    典型使用流程：
    1. setup_data(): 加载数据配置，创建数据源，构建特征图并拟合
    2. setup_model(): 从特征图推断输入维度，创建模型实例
    3. save_pipeline_state(): 保存完整的 pipeline 状态用于后续验证和恢复
    """

    def __init__(self, pipeline_config_path: str):
        """
        初始化 Pipeline 编排器。

        Args:
            pipeline_config_path: pipeline.yaml 配置文件的路径
        """
        self.pipeline_config_path = Path(pipeline_config_path)
        self.config = self._load_config()
        self.data_config = None
        self.model_config = None
        self.train_config = None
        self.feature_graph = None
        self.expression_engine = None
        self.pipeline_state: PipelineState | None = None

    def _load_config(self) -> dict[str, Any]:
        """加载 pipeline.yaml 配置。"""
        if not self.pipeline_config_path.exists():
            raise FileNotFoundError(f"Pipeline config not found: {self.pipeline_config_path}")

        with open(self.pipeline_config_path) as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded pipeline config from {self.pipeline_config_path}")
        return config

    def setup_data(self, fit_features: bool = True, fit_samples: int = 10000) -> tuple[HEPDataset, HEPDataset | None]:
        """
        设置数据系统：加载数据配置、创建数据源、构建特征图并创建数据集。

        如果 fit_features=True，会从训练数据中采样 fit_samples 个样本用于拟合特征处理器
        （计算 Normalizer 的统计参数）。这对于确保训练/验证/测试数据的一致性至关重要。

        验证集分割：
        - 如果配置了 val_split > 0 且提供了 load_range，会自动进行数据分割
        - 验证集使用独立的数据源，确保数据不重叠

        Args:
            fit_features: 是否拟合特征处理器（计算 Normalizer 参数）
            fit_samples: 用于拟合的样本数量

        Returns:
            (训练数据集, 验证数据集) 元组，验证数据集可能为 None
        """
        # 加载 DataConfig
        data_config_path = self.config.get("data", {}).get("config_path")
        if data_config_path:
            self.data_config = DataConfig.load(data_config_path)
        else:
            # 从 pipeline.yaml 直接读取 data 配置
            data_dict = self.config.get("data", {})
            self.data_config = DataConfig(**data_dict)

        # 创建数据源
        data_source_path = self.config.get("data", {}).get("source_path")
        if not data_source_path:
            raise ValueError("data.source_path must be specified in pipeline.yaml")

        treename = self.config.get("data", {}).get("treename", "events")

        # 优先读取 train_range/val_range/test_range（与 pipeline_edm4hep.yaml 一致），
        # 回退到旧的 load_range + val_split 以保持向后兼容
        data_section = self.config.get("data", {})
        train_range_cfg = data_section.get("train_range")
        val_range_cfg = data_section.get("val_range")
        test_range_cfg = data_section.get("test_range")
        has_new_range_style = train_range_cfg is not None

        # 旧式兼容
        load_range = data_section.get("load_range")
        val_split = data_section.get("val_split", 0.0)

        # 解析字典格式的数据路径（类似 weaver 的 to_filelist）
        # 如果 data_source_path 是字典格式（包含 label:path），会自动生成 file_magic 和 labels
        import glob
        import os
        import re

        # 解析数据路径
        if isinstance(data_source_path, str):
            path_list = data_source_path.split()
        else:
            path_list = list(data_source_path) if data_source_path else []

        file_dict = {}
        has_dict_format = False
        all_files = []

        for path_item in path_list:
            if ":" in path_item and not path_item.startswith("/") and not path_item.startswith("."):
                # 字典格式：label:path
                has_dict_format = True
                parts = path_item.split(":", 1)
                if len(parts) == 2:
                    label, file_path = parts
                    files = glob.glob(file_path)
                    if label in file_dict:
                        file_dict[label] += files
                    else:
                        file_dict[label] = files
                    all_files.extend(files)
            else:
                # 普通路径格式
                files = glob.glob(path_item)
                if "_" in file_dict:
                    file_dict["_"] += files
                else:
                    file_dict["_"] = files
                all_files.extend(files)

        all_files = sorted(list(set(all_files)))

        # 如果使用字典方式，生成 file_magic 和 labels
        auto_file_magic = None
        auto_labels = None

        if has_dict_format and "_" not in file_dict:
            auto_file_magic = {}
            auto_labels = []

            for label, files in file_dict.items():
                label_var = f"is_{label}"
                auto_labels.append(label_var)

                # 为每个文件生成匹配模式（使用目录名或文件名）
                patterns = {}
                for file_path in files:
                    dir_name = os.path.basename(os.path.dirname(file_path))
                    if dir_name:
                        pattern = re.escape(dir_name)
                        patterns[pattern] = 1.0
                    else:
                        file_name = os.path.basename(file_path)
                        if file_name:
                            pattern = re.escape(os.path.splitext(file_name)[0])
                            patterns[pattern] = 1.0

                if not patterns:
                    patterns[re.escape(label)] = 1.0

                auto_file_magic[label_var] = patterns

            # 如果自动生成了 labels，更新 DataConfig 和模型配置
            if auto_labels and len(auto_labels) > 0:
                # 更新模型配置中的 num_classes
                model_config = self.config.get("model", {})
                model_params = model_config.get("params", {})
                if "num_classes" not in model_params or model_params["num_classes"] is None:
                    model_params["num_classes"] = len(auto_labels)
                    logger.info(f"Auto-detected num_classes={len(auto_labels)} from data paths")

                # 更新 DataConfig 的 labels 配置
                if self.data_config and (not self.data_config.label_names or len(self.data_config.label_names) == 0):
                    self.data_config.label_type = "simple"
                    self.data_config.label_value = auto_labels
                    self.data_config.label_names = ("_label_",)
                    # 构建标签表达式
                    label_exprs = [f"ak.to_numpy({k})" for k in auto_labels]
                    self.data_config.register(
                        "_label_",
                        f"np.argmax(np.stack([{','.join(label_exprs)}], axis=1), axis=1)",
                    )
                    self.data_config.register(
                        "_labelcheck_",
                        f"np.sum(np.stack([{','.join(label_exprs)}], axis=1), axis=1)",
                        "train",
                    )
                    logger.info(f"Auto-generated labels from data paths: {auto_labels}")

        # 使用解析后的文件列表
        if all_files:
            data_source_path = all_files

        # 处理验证集分割
        if has_new_range_style:
            # 新式：直接使用 train_range / val_range / test_range
            train_range = train_range_cfg
            val_range = val_range_cfg
            # test_range_cfg 暂存，由调用方（如 run_pipeline.py）按需使用
            logger.info(f"Using new-style ranges: train={train_range}, val={val_range}, test={test_range_cfg}")
        else:
            # 旧式：load_range + val_split
            train_range = load_range
            val_range = None

            if val_split > 0:
                if load_range:
                    start, end = load_range
                    total = end - start
                    split_point = int(start + total * (1 - val_split))
                    train_range = [start, split_point]
                    val_range = [split_point, end]
                    logger.info(f"Splitting data into train range {train_range} and val range {val_range} (split={val_split})")
                else:
                    raise ValueError(
                        "val_split specified but load_range is missing. "
                        "Cannot safely split data without knowing the total data size. "
                        "To fix this, either:\n"
                        "  1. Specify load_range in data config (e.g., [0.0, 1.0] for all data)\n"
                        "  2. Use separate data files for training and validation\n"
                        "  3. Set val_split=0.0 to disable validation split"
                    )

        # ---------------------------------------------------------------
        # Classes-based label system: resolve file globs → class_labels
        # ---------------------------------------------------------------
        class_labels = None
        if self.data_config.class_names is not None:
            class_labels = {}
            class_all_files = []
            for idx, (name, pattern) in enumerate(zip(self.data_config.class_names, self.data_config.class_files)):
                matched = sorted(glob.glob(pattern))
                if not matched:
                    raise ValueError(f"No files matched for class '{name}' with pattern '{pattern}'")
                for fp in matched:
                    abs_fp = os.path.abspath(fp)
                    if abs_fp in class_labels:
                        raise ValueError(f"File '{fp}' matches multiple classes. " f"Each file must belong to exactly one class.")
                    class_labels[abs_fp] = idx
                    class_all_files.append(abs_fp)
                logger.info(f"Class '{name}' (index {idx}): {len(matched)} file(s)")

            # Override data_source_path — files come from classes definitions
            data_source_path = class_all_files
            logger.info(f"Classes-based labels: {len(class_labels)} files across " f"{len(self.data_config.class_names)} classes")

        data_source = DataSourceFactory.create(
            data_source_path,
            treename=treename,
            load_range=train_range,
            file_magic=self.data_config.file_magic,
            class_labels=class_labels,
        )

        # 设置特征系统（必需）
        # 新式: data.features_config, 旧式 fallback: features.config_path
        feature_config_path = self.config.get("data", {}).get("features_config") or self.config.get("features", {}).get("config_path")
        if not feature_config_path:
            raise ValueError("features.config_path is required. Feature definitions must be in features.yaml via FeatureGraph.")

        if self.feature_graph is None:
            self.expression_engine = ExpressionEngine()
            self.feature_graph = FeatureGraph.from_yaml(
                feature_config_path,
                self.expression_engine,
            )

        # 拟合特征（如果需要）
        if fit_features:
            logger.info(f"Fitting features with {fit_samples} samples...")
            # 临时加载一部分数据用于拟合
            # 注意：我们需要加载 FeatureGraph 需要的所有原始分支
            # 为了简单起见，我们加载 DataConfig 中指定的所有训练分支
            load_branches = list(self.data_config.train_load_branches)
            # Also include FeatureGraph source branches needed for fitting
            if self.feature_graph is not None:
                fg_branches = list(self.feature_graph.get_source_branches())
                load_branches = list(set(load_branches + fg_branches))

            # 使用 data_source 的切片功能（如果支持）或加载后切片
            # 这里假设 load_branches 返回所有数据，我们只取前 fit_samples
            # 更好的做法是在 data_source 层面支持 limit，但现在先这样
            raw_table = data_source.load_branches(load_branches)

            if len(raw_table) > fit_samples:
                raw_table = raw_table[:fit_samples]

            self.feature_graph.fit(raw_table)

        # 创建训练数据集
        train_dataset = HEPDataset(
            data_source=data_source,
            data_config=self.data_config,
            feature_graph=self.feature_graph,  # 必需参数
            for_training=True,
            shuffle=True,
        )

        # 创建验证数据集（如果有配置）
        val_dataset = None
        if val_range is not None:
            # 创建验证集数据源（使用 val_range）
            val_data_source = DataSourceFactory.create(
                data_source_path,
                treename=treename,
                load_range=val_range,
                file_magic=self.data_config.file_magic,
                class_labels=class_labels,
            )

            val_dataset = HEPDataset(
                data_source=val_data_source,
                data_config=self.data_config,
                feature_graph=self.feature_graph,
                for_training=False,  # 使用测试配置（无 shuffle/reweight）
                shuffle=False,
            )
            logger.info(f"Validation dataset created with range {val_range}")

        logger.info("Data system setup complete")
        return train_dataset, val_dataset

    def setup_model(self) -> Any:
        """
        设置模型系统：从配置创建模型实例。

        统一使用新架构（event/object 多输入），自动从 `FeatureGraph.output_spec()` 推断维度：
        - 自动推断 `event_input_dim`（如果 FeatureGraph 有 event-level 特征）
        - 自动推断 `object_input_dim`（如果 FeatureGraph 有 object-level 特征）
        - `embed_dim` 需要在配置中指定，或使用默认值 64（会发出警告）

        该方法会验证配置中的维度参数（如果存在）与推断值的一致性，
        如果不一致，会使用推断值并发出警告。

        Returns:
            创建的模型实例

        Raises:
            ValueError: 如果 feature_graph 未初始化或无法推断输入维度
        """
        model_config = self.config.get("model", {})
        if not model_config:
            raise ValueError("model config must be specified in pipeline.yaml")

        model_name = model_config.get("name")
        if not model_name:
            # 新式 YAML 无 model.name — 从 task_type 推断
            task_type = self.get_train_config().get("task_type", "classification")
            model_name = "mlp_classifier" if task_type == "classification" else "mlp_regressor"
            logger.info(f"model.name not specified; inferred '{model_name}' from task_type='{task_type}'")

        # 获取模型参数: 优先使用 model.params（旧式），否则把 model 下除 name/params 外的
        # 所有键当作模型参数（新式平铺写法）
        model_kwargs = model_config.get("params", {}).copy()
        if not model_kwargs:
            # 新式平铺：model 节下除 name/params 外的所有键视为模型参数
            model_kwargs = {k: v for k, v in model_config.items() if k not in ("name", "params")}

        # 从 FeatureGraph.output_spec() 推断输入维度（统一使用新架构）
        if self.feature_graph is None:
            raise ValueError("feature_graph is required to infer input dimensions. Call setup_data() first.")

        output_spec = self.feature_graph.output_spec()

        # 自动推断 event_input_dim 和 object_input_dim
        has_event_input_dim = "event_input_dim" in model_kwargs
        has_object_input_dim = "object_input_dim" in model_kwargs

        if "event" in output_spec:
            inferred_event_dim = output_spec["event"]["dim"]
            if not has_event_input_dim:
                model_kwargs["event_input_dim"] = inferred_event_dim
                logger.info(f"Auto-inferred event_input_dim={inferred_event_dim} from FeatureGraph.output_spec()")
            elif model_kwargs["event_input_dim"] != inferred_event_dim:
                logger.warning(
                    f"event_input_dim mismatch: config has {model_kwargs['event_input_dim']}, "
                    f"but FeatureGraph.output_spec() suggests {inferred_event_dim}. Using {inferred_event_dim}."
                )
                model_kwargs["event_input_dim"] = inferred_event_dim
        elif has_event_input_dim:
            # 如果配置中指定了 event_input_dim，但 output_spec 中没有 event，发出警告
            logger.warning(
                f"event_input_dim={model_kwargs['event_input_dim']} specified in config, "
                "but FeatureGraph.output_spec() has no event-level features. Removing event_input_dim."
            )
            model_kwargs.pop("event_input_dim", None)

        if "object" in output_spec:
            inferred_object_dim = output_spec["object"]["dim"]
            if not has_object_input_dim:
                model_kwargs["object_input_dim"] = inferred_object_dim
                logger.info(f"Auto-inferred object_input_dim={inferred_object_dim} from FeatureGraph.output_spec()")
            elif model_kwargs["object_input_dim"] != inferred_object_dim:
                logger.warning(
                    f"object_input_dim mismatch: config has {model_kwargs['object_input_dim']}, "
                    f"but FeatureGraph.output_spec() suggests {inferred_object_dim}. Using {inferred_object_dim}."
                )
                model_kwargs["object_input_dim"] = inferred_object_dim
        elif has_object_input_dim:
            # 如果配置中指定了 object_input_dim，但 output_spec 中没有 object，发出警告
            logger.warning(
                f"object_input_dim={model_kwargs['object_input_dim']} specified in config, "
                "but FeatureGraph.output_spec() has no object-level features. Removing object_input_dim."
            )
            model_kwargs.pop("object_input_dim", None)

        # 确保至少有一个输入维度
        if "event_input_dim" not in model_kwargs and "object_input_dim" not in model_kwargs:
            raise ValueError("No input dimensions found. FeatureGraph.output_spec() must contain at least 'event' or 'object' features.")

        # 确保 embed_dim 已提供（新架构必需）
        if "embed_dim" not in model_kwargs:
            logger.warning("embed_dim not specified. Using default value 64. " "Please specify embed_dim explicitly for better control.")
            model_kwargs["embed_dim"] = 64

        # 自动推断 num_classes（分类任务）
        task_type = self.get_train_config().get("task_type", "classification")
        if task_type == "classification" and "num_classes" not in model_kwargs:
            if hasattr(self, "data_config") and self.data_config is not None and self.data_config.class_names is not None:
                num_classes = len(self.data_config.class_names)
                model_kwargs["num_classes"] = num_classes
                logger.info(f"Auto-inferred num_classes={num_classes} from " f"data_config.class_names={self.data_config.class_names}")
            else:
                logger.warning(
                    "Classification task but num_classes not specified and " "data_config.class_names not available. Model will use its default."
                )

        # 创建模型
        model = get_model(model_name, **model_kwargs)

        # Normalize model_config to always have 'name' + 'params' structure
        # (flat config style puts params at top level without a 'params' key)
        self.model_config = {
            "name": model_name,
            "params": model_kwargs,
        }
        logger.info(f"Model '{model_name}' created with params: {model_kwargs}")

        # 创建并验证 PipelineState
        task_type = self.get_train_config().get("task_type", "classification")
        self.pipeline_state = PipelineState.from_configs(
            feature_graph=self.feature_graph,
            model_config=self.model_config,
            task_type=task_type,
            pipeline_config_path=str(self.pipeline_config_path),
            feature_config_path=self.config.get("data", {}).get("features_config") or self.config.get("features", {}).get("config_path"),
            data_config_path=self.config.get("data", {}).get("config_path"),
        )

        # 验证配置一致性
        is_valid, errors = self.pipeline_state.validate()
        if not is_valid:
            error_msg = "Pipeline configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("Pipeline state validated successfully")
        logger.info(f"Pipeline state: {self.pipeline_state.get_model_info()}")

        return model

    def get_train_config(self) -> dict[str, Any]:
        """
        获取训练配置。

        Returns:
            训练配置字典
        """
        # 新式: "training", 旧式 fallback: "train"
        train_config = self.config.get("training", {}) or self.config.get("train", {})
        self.train_config = train_config
        return train_config

    def get_config(self) -> dict[str, Any]:
        """
        获取完整配置。

        Returns:
            完整配置字典
        """
        return self.config

    def get_data_config(self) -> DataConfig | None:
        """获取 DataConfig。"""
        return self.data_config

    def get_model_config(self) -> dict[str, Any] | None:
        """获取模型配置。"""
        return self.model_config

    def get_feature_graph(self) -> FeatureGraph | None:
        """获取特征图。"""
        return self.feature_graph

    def get_pipeline_state(self) -> PipelineState | None:
        """
        获取 PipelineState。

        Returns:
            PipelineState: Pipeline 状态（如果已创建）
        """
        return self.pipeline_state

    def save_pipeline_state(self, path: str | Path) -> None:
        """
        保存 PipelineState 到文件。

        Args:
            path: 保存路径
        """
        if self.pipeline_state is None:
            raise ValueError("PipelineState not created. Call setup_model() first.")
        self.pipeline_state.save(path)
