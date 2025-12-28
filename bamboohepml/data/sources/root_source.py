"""
ROOT 数据源实现
"""
import math
import awkward as ak
from typing import List, Optional

from .base import DataSource, DataSourceConfig
from ..logger import _logger


class ROOTDataSource(DataSource):
    """ROOT 文件数据源。"""
    
    def load_branches(self, branches: List[str]) -> ak.Array:
        """加载指定的分支。
        
        Args:
            branches (List[str]): 要加载的分支列表
            
        Returns:
            ak.Array: 加载的数据
        """
        import uproot
        from ..tools import _concat
        import re
        
        table = []
        branches = list(branches)
        
        for filepath in self._file_paths:
            try:
                with uproot.open(filepath) as f:
                    # 确定树名
                    treename = self.config.treename
                    if treename is None:
                        treenames = set([
                            k.split(';')[0] for k, v in f.items()
                            if getattr(v, 'classname', '') == 'TTree'
                        ])
                        if len(treenames) == 1:
                            treename = treenames.pop()
                        else:
                            raise RuntimeError(
                                f'需要指定 `treename`，因为文件 {filepath} 中找到多个树: {treenames}')
                    
                    tree = f[treename]
                    
                    # 处理加载范围
                    load_range = self.config.load_range
                    if load_range is not None:
                        start = math.trunc(load_range[0] * tree.num_entries)
                        stop = max(start + 1, math.trunc(load_range[1] * tree.num_entries))
                    else:
                        start, stop = None, None
                    
                    # 处理分支名称映射
                    branch_magic = self.config.branch_magic
                    if branch_magic is not None:
                        branch_dict = {}
                        for name in branches:
                            decoded_name = name
                            for src, tgt in branch_magic.items():
                                if src in decoded_name:
                                    decoded_name = decoded_name.replace(src, tgt)
                            branch_dict[name] = decoded_name
                        outputs = tree.arrays(
                            filter_name=list(branch_dict.values()),
                            entry_start=start,
                            entry_stop=stop
                        )
                        for name, decoded_name in branch_dict.items():
                            if name != decoded_name:
                                outputs[name] = outputs[decoded_name]
                    else:
                        outputs = tree.arrays(
                            filter_name=branches,
                            entry_start=start,
                            entry_stop=stop
                        )
                    
                    # 处理 file_magic
                    if self.config.file_magic is not None:
                        for var, value_dict in self.config.file_magic.items():
                            if var in outputs.fields:
                                _logger.warning(
                                    f'变量 `{var}` 已在数组中定义，但将被 file_magic {value_dict} 覆盖。')
                            outputs[var] = 0
                            for fn_pattern, value in value_dict.items():
                                if re.search(fn_pattern, filepath):
                                    outputs[var] = value
                                    break
                    
                    table.append(outputs)
            except Exception as e:
                _logger.error(f'读取文件 {filepath} 时出错: {e}')
                import traceback
                _logger.error(traceback.format_exc())
        
        if len(table) == 0:
            raise RuntimeError(f'从文件列表 {self._file_paths} 加载了零条记录。')
        
        return _concat(table)
    
    def get_available_branches(self) -> List[str]:
        """获取可用的分支列表。
        
        Returns:
            List[str]: 可用分支列表
        """
        import uproot
        
        if len(self._file_paths) == 0:
            return []
        
        # 使用第一个文件获取分支列表
        filepath = self._file_paths[0]
        try:
            with uproot.open(filepath) as f:
                treename = self.config.treename
                if treename is None:
                    treenames = set([
                        k.split(';')[0] for k, v in f.items()
                        if getattr(v, 'classname', '') == 'TTree'
                    ])
                    if len(treenames) == 1:
                        treename = treenames.pop()
                    else:
                        return []
                
                tree = f[treename]
                return list(tree.keys())
        except Exception as e:
            _logger.error(f'获取分支列表时出错: {e}')
            return []
    
    def get_num_events(self) -> Optional[int]:
        """获取事件数量。
        
        Returns:
            int: 事件数量
        """
        import uproot
        
        total_events = 0
        for filepath in self._file_paths:
            try:
                with uproot.open(filepath) as f:
                    treename = self.config.treename
                    if treename is None:
                        treenames = set([
                            k.split(';')[0] for k, v in f.items()
                            if getattr(v, 'classname', '') == 'TTree'
                        ])
                        if len(treenames) == 1:
                            treename = treenames.pop()
                        else:
                            continue
                    
                    tree = f[treename]
                    total_events += tree.num_entries
            except Exception:
                continue
        
        return total_events if total_events > 0 else None

