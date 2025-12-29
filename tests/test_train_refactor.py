"""
训练任务重构后的测试

简化的测试，只关注关键流程：
1. train_task 的基本工作流程
2. LocalBackend 和 RayBackend 的选择
3. 关键步骤的调用顺序
"""

import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock Ray for testing (Ray is optional dependency)
sys.modules["ray"] = MagicMock()
sys.modules["ray.data"] = MagicMock()
sys.modules["ray.train"] = MagicMock()
sys.modules["ray.train.torch"] = MagicMock()
sys.modules["ray.train.torch"].TorchTrainer = MagicMock()

from bamboohepml.tasks.train import train_task  # noqa: E402


class TestTrainTaskRefactor(unittest.TestCase):
    """测试训练任务重构后的流程"""

    def setUp(self):
        """设置测试环境"""
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, "pipeline.yaml")

        # 创建虚拟配置文件
        with open(self.config_path, "w") as f:
            f.write("dummy: config")

    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.test_dir)

    def _setup_orchestrator_mocks(self, mock_orchestrator):
        """设置 orchestrator 的基本 mock，返回可序列化的数据"""
        # 基本配置
        mock_orchestrator.get_train_config.return_value = {
            "num_epochs": 5,
            "batch_size": 32,
            "learning_rate": 0.001,
            "task_type": "classification",
            "learning_paradigm": "supervised",
        }
        mock_orchestrator.get_input_dim_from_spec.return_value = 10
        mock_orchestrator.config = {"model": {"name": "test_model", "params": {}}}
        mock_orchestrator.get_model_config.return_value = {"name": "test_model", "params": {}}

        # setup_data 返回 (train_dataset, val_dataset)
        mock_train_dataset = MagicMock()
        mock_orchestrator.setup_data.return_value = (mock_train_dataset, None)

        # setup_model 返回模型
        mock_orchestrator.setup_model.return_value = MagicMock(spec=torch.nn.Module)

        # feature_graph 返回可序列化的字典
        mock_feature_graph = MagicMock()
        mock_feature_graph.output_spec.return_value = {}
        mock_feature_graph.export_state.return_value = {}
        mock_orchestrator.feature_graph = mock_feature_graph
        mock_orchestrator.get_feature_graph.return_value = mock_feature_graph

        # pipeline_state
        mock_pipeline_state = MagicMock()
        mock_pipeline_state.input_dim = 10
        mock_pipeline_state.input_key = "event"
        mock_orchestrator.pipeline_state = mock_pipeline_state
        mock_orchestrator.get_pipeline_state.return_value = mock_pipeline_state

        # save_pipeline_state 方法
        mock_orchestrator.save_pipeline_state = MagicMock()

        return mock_train_dataset

    @patch("bamboohepml.tasks.train.PipelineOrchestrator")
    @patch("torch.utils.data.DataLoader")
    @patch("bamboohepml.tasks.train.Trainer")
    @patch("torch.save")
    def test_train_task_local_backend(self, mock_torch_save, mock_trainer_cls, mock_dataloader_cls, mock_orchestrator_cls):
        """测试 train_task 使用 LocalBackend 的基本流程"""
        # 设置 mocks
        mock_orchestrator = mock_orchestrator_cls.return_value
        self._setup_orchestrator_mocks(mock_orchestrator)

        # Trainer mock
        mock_trainer = mock_trainer_cls.return_value
        mock_trainer.fit.return_value = {"loss": [0.1, 0.05]}

        # 执行
        output_dir = os.path.join(self.test_dir, "output")
        train_task(
            pipeline_config_path=self.config_path,
            experiment_name="test",
            use_ray=False,
            output_dir=output_dir,
        )

        # 验证关键流程
        # 1. Orchestrator 初始化
        mock_orchestrator_cls.assert_called_once_with(self.config_path)

        # 2. 数据设置
        mock_orchestrator.setup_data.assert_called_once_with(fit_features=True)

        # 3. 获取训练配置
        mock_orchestrator.get_train_config.assert_called()

        # 4. 模型设置
        mock_orchestrator.setup_model.assert_called_once()

        # 5. Trainer 初始化（验证关键参数）
        mock_trainer_cls.assert_called_once()
        call_kwargs = mock_trainer_cls.call_args[1]
        self.assertEqual(call_kwargs["task_type"], "classification")
        self.assertEqual(call_kwargs["learning_paradigm"], "supervised")

        # 6. 训练执行
        mock_trainer.fit.assert_called_once()
        self.assertEqual(mock_trainer.fit.call_args[1]["num_epochs"], 5)
        self.assertEqual(mock_trainer.fit.call_args[1]["save_dir"], output_dir)

        # 7. 保存状态
        mock_orchestrator.save_pipeline_state.assert_called()
        mock_torch_save.assert_called()

    @patch("bamboohepml.tasks.train.PipelineOrchestrator")
    @patch("bamboohepml.tasks.train.ray")
    @patch("bamboohepml.tasks.train.TorchTrainer")
    def test_train_task_ray_backend(self, mock_torch_trainer_cls, mock_ray, mock_orchestrator_cls):
        """测试 train_task 使用 RayBackend 的基本流程"""
        # Mock Ray 可用
        with patch("bamboohepml.tasks.train.RAY_AVAILABLE", True):
            # 设置 mocks
            mock_orchestrator = mock_orchestrator_cls.return_value
            self._setup_orchestrator_mocks(mock_orchestrator)

            # Ray mock
            mock_ray.is_initialized.return_value = False

            # TorchTrainer mock
            mock_torch_trainer = mock_torch_trainer_cls.return_value
            mock_result = MagicMock()
            mock_result.best_checkpoints = []
            mock_torch_trainer.fit.return_value = mock_result

            # 执行
            train_task(
                pipeline_config_path=self.config_path,
                use_ray=True,
                num_workers=2,
                gpu_per_worker=0,
            )

            # 验证关键流程
            # 1. Ray 初始化
            mock_ray.init.assert_called_once()

            # 2. TorchTrainer 初始化
            mock_torch_trainer_cls.assert_called_once()
            call_kwargs = mock_torch_trainer_cls.call_args[1]

            # 验证关键配置传递
            train_config = call_kwargs["train_loop_config"]
            self.assertEqual(train_config["task_type"], "classification")
            self.assertEqual(train_config["num_epochs"], 5)

            # 验证必要的配置项存在
            self.assertIn("scaling_config", call_kwargs)
            self.assertIn("run_config", call_kwargs)
            self.assertIn("datasets", call_kwargs)

            # 3. 训练执行
            mock_torch_trainer.fit.assert_called_once()

    @patch("bamboohepml.tasks.train.PipelineOrchestrator")
    @patch("torch.utils.data.DataLoader")
    @patch("bamboohepml.tasks.train.Trainer")
    def test_train_task_config_override(self, mock_trainer_cls, mock_dataloader_cls, mock_orchestrator_cls):
        """测试 train_task 的参数覆盖功能"""
        # 设置 mocks
        mock_orchestrator = mock_orchestrator_cls.return_value
        self._setup_orchestrator_mocks(mock_orchestrator)

        mock_trainer = mock_trainer_cls.return_value
        mock_trainer.fit.return_value = {"loss": [0.1]}

        # 执行，覆盖配置
        train_task(
            pipeline_config_path=self.config_path,
            num_epochs=10,  # 覆盖默认值
            batch_size=64,  # 覆盖默认值
            learning_rate=0.01,  # 覆盖默认值
            use_ray=False,
        )

        # 验证配置被覆盖
        train_config = mock_orchestrator.get_train_config.return_value
        self.assertEqual(train_config["num_epochs"], 10)
        self.assertEqual(train_config["batch_size"], 64)
        self.assertEqual(train_config["learning_rate"], 0.01)

        # 验证 Trainer.fit 使用了覆盖后的配置
        mock_trainer.fit.assert_called_once()
        self.assertEqual(mock_trainer.fit.call_args[1]["num_epochs"], 10)


if __name__ == "__main__":
    unittest.main()
