"""
ONNX 推理接口

提供 ONNX 模型的推理功能。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

try:
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from ..config import logger


class ONNXPredictor:
    """
    ONNX 预测器

    使用 ONNX Runtime 进行推理。
    """

    def __init__(self, onnx_path: str, providers: Optional[List[str]] = None):
        """
        初始化 ONNX 预测器。

        Args:
            onnx_path: ONNX 模型文件路径
            providers: 执行提供者列表（如 ['CUDAExecutionProvider', 'CPUExecutionProvider']）
        """
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime is not installed. Install with: pip install onnxruntime")

        self.onnx_path = onnx_path

        # 设置执行提供者
        if providers is None:
            providers = ["CPUExecutionProvider"]
            # 如果有 CUDA，优先使用
            try:
                ort.InferenceSession("", providers=["CUDAExecutionProvider"])
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            except Exception:
                pass

        # 创建推理会话
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=providers,
        )

        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        logger.info(f"ONNX model loaded from {onnx_path}")
        logger.info(f"Input: {self.input_name}, Output: {self.output_name}")
        logger.info(f"Providers: {providers}")

    def predict(
        self,
        features: np.ndarray,
        return_probabilities: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        预测。

        Args:
            features: 特征数组（shape: [batch_size, features]）
            return_probabilities: 是否返回概率（仅分类任务）

        Returns:
            预测结果列表
        """
        # 确保是 numpy 数组
        if not isinstance(features, np.ndarray):
            features = np.array(features, dtype=np.float32)

        # 确保是 float32
        if features.dtype != np.float32:
            features = features.astype(np.float32)

        # 运行推理
        ort_inputs = {self.input_name: features}
        ort_outs = self.session.run([self.output_name], ort_inputs)
        outputs = ort_outs[0]

        # 处理输出
        results = []
        for i in range(len(outputs)):
            output = outputs[i]

            if len(output.shape) == 0:
                # 标量输出（回归）
                prediction = float(output)
                result = {"prediction": prediction}
            elif len(output) > 1:
                # 向量输出（分类）
                prediction = int(np.argmax(output))
                result = {"prediction": prediction}
                if return_probabilities:
                    # 计算 softmax（如果输出不是概率）
                    probs = self._softmax(output)
                    result["probabilities"] = probs.tolist()
            else:
                # 单值输出（回归）
                prediction = float(output[0])
                result = {"prediction": prediction}

            results.append(result)

        return results

    def predict_batch(
        self,
        features_list: List[List[float]],
        return_probabilities: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        批量预测。

        Args:
            features_list: 特征列表
            return_probabilities: 是否返回概率

        Returns:
            预测结果列表
        """
        features_array = np.array(features_list, dtype=np.float32)
        return self.predict(features_array, return_probabilities=return_probabilities)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """计算 softmax。"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息。"""
        input_info = self.session.get_inputs()[0]
        output_info = self.session.get_outputs()[0]

        return {
            "onnx_path": self.onnx_path,
            "input_name": input_info.name,
            "input_shape": input_info.shape,
            "input_type": input_info.type,
            "output_name": output_info.name,
            "output_shape": output_info.shape,
            "output_type": output_info.type,
        }
