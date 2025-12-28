"""
Parquet 数据源实现
"""

import math

import awkward as ak

from ..logger import _logger
from .base import DataSource


class ParquetDataSource(DataSource):
    """Parquet 文件数据源。"""

    def load_branches(self, branches: list[str]) -> ak.Array:
        """加载指定的分支。

        Args:
            branches (List[str]): 要加载的分支列表

        Returns:
            ak.Array: 加载的数据
        """
        import re

        from ..tools import _concat

        table = []
        branches = list(branches)

        for filepath in self._file_paths:
            try:
                outputs = ak.from_parquet(filepath, columns=branches)

                # 处理加载范围
                load_range = self.config.load_range
                if load_range is not None:
                    start = math.trunc(load_range[0] * len(outputs))
                    stop = max(start + 1, math.trunc(load_range[1] * len(outputs)))
                    outputs = outputs[start:stop]

                # 处理 file_magic
                if self.config.file_magic is not None:
                    for var, value_dict in self.config.file_magic.items():
                        if var in outputs.fields:
                            _logger.warning(f"变量 `{var}` 已在数组中定义，但将被 file_magic {value_dict} 覆盖。")
                        outputs[var] = 0
                        for fn_pattern, value in value_dict.items():
                            if re.search(fn_pattern, filepath):
                                outputs[var] = value
                                break

                table.append(outputs)
            except Exception as e:
                _logger.error(f"读取文件 {filepath} 时出错: {e}")
                import traceback

                _logger.error(traceback.format_exc())

        if len(table) == 0:
            raise RuntimeError(f"从文件列表 {self._file_paths} 加载了零条记录。")

        return _concat(table)

    def get_available_branches(self) -> list[str]:
        """获取可用的分支列表。

        Returns:
            List[str]: 可用分支列表
        """
        if len(self._file_paths) == 0:
            return []

        # 使用第一个文件获取分支列表
        filepath = self._file_paths[0]
        try:
            # 读取元数据获取列名
            import pyarrow.parquet as pq

            parquet_file = pq.ParquetFile(filepath)
            return list(parquet_file.schema.names)
        except Exception as e:
            _logger.error(f"获取分支列表时出错: {e}")
            return []

    def get_num_events(self) -> int | None:
        """获取事件数量。

        Returns:
            int: 事件数量
        """
        total_events = 0
        for filepath in self._file_paths:
            try:
                import pyarrow.parquet as pq

                parquet_file = pq.ParquetFile(filepath)
                total_events += parquet_file.metadata.num_rows
            except Exception:
                continue

        return total_events if total_events > 0 else None
