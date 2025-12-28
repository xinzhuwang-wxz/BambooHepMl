"""
HDF5 数据源实现
"""

import math
from typing import Optional

import awkward as ak

from ..logger import _logger
from .base import DataSource


class HDF5DataSource(DataSource):
    """HDF5 文件数据源。"""

    def load_branches(self, branches: list[str]) -> ak.Array:
        """加载指定的分支。

        Args:
            branches (List[str]): 要加载的分支列表

        Returns:
            ak.Array: 加载的数据
        """
        import re

        import tables

        from ..tools import _concat

        tables.set_blosc_max_threads(4)
        table = []
        branches = list(branches)

        for filepath in self._file_paths:
            try:
                with tables.open_file(filepath) as f:
                    outputs = {k: getattr(f.root, k)[:] for k in branches}

                # 处理加载范围
                load_range = self.config.load_range
                if load_range is None:
                    load_range = (0, 1)
                start = math.trunc(load_range[0] * len(outputs[branches[0]]))
                stop = max(start + 1, math.trunc(load_range[1] * len(outputs[branches[0]])))
                for k, v in outputs.items():
                    outputs[k] = v[start:stop]

                # 处理 file_magic
                if self.config.file_magic is not None:
                    for var, value_dict in self.config.file_magic.items():
                        if var in outputs:
                            _logger.warning(f"变量 `{var}` 已在数组中定义，但将被 file_magic {value_dict} 覆盖。")
                        outputs[var] = 0
                        for fn_pattern, value in value_dict.items():
                            if re.search(fn_pattern, filepath):
                                outputs[var] = value
                                break

                table.append(ak.Array(outputs))
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
        import tables

        if len(self._file_paths) == 0:
            return []

        # 使用第一个文件获取分支列表
        filepath = self._file_paths[0]
        try:
            with tables.open_file(filepath) as f:
                return list(f.root._v_children.keys())
        except Exception as e:
            _logger.error(f"获取分支列表时出错: {e}")
            return []

    def get_num_events(self) -> Optional[int]:
        """获取事件数量。

        Returns:
            int: 事件数量
        """
        import tables

        total_events = 0
        for filepath in self._file_paths:
            try:
                with tables.open_file(filepath) as f:
                    # 使用第一个分支获取长度
                    if len(f.root._v_children) > 0:
                        first_branch = list(f.root._v_children.keys())[0]
                        total_events += len(getattr(f.root, first_branch))
            except Exception:
                continue

        return total_events if total_events > 0 else None
