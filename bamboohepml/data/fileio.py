"""
文件 I/O 模块

支持多种 HEP 数据格式：
- ROOT (.root)
- HDF5 (.h5)
- Parquet (.parquet)
- Awkward (.awkd)
"""

import math
import traceback

import awkward as ak
import tqdm

from .logger import _logger, warn_n_times
from .tools import _concat


def _read_hdf5(filepath, branches, load_range=None):
    """读取 HDF5 文件。

    Args:
        filepath (str): 文件路径。
        branches (list): 要读取的分支列表。
        load_range (tuple, optional): 加载范围 (start, end)，范围在 [0, 1]。默认为 None。

    Returns:
        ak.Array: 读取的数据。
    """
    import tables

    tables.set_blosc_max_threads(4)
    with tables.open_file(filepath) as f:
        outputs = {k: getattr(f.root, k)[:] for k in branches}
    if load_range is None:
        load_range = (0, 1)
    start = math.trunc(load_range[0] * len(outputs[branches[0]]))
    stop = max(start + 1, math.trunc(load_range[1] * len(outputs[branches[0]])))
    for k, v in outputs.items():
        outputs[k] = v[start:stop]
    return ak.Array(outputs)


def _read_root(filepath, branches, load_range=None, treename=None, branch_magic=None):
    """读取 ROOT 文件。

    Args:
        filepath (str): 文件路径。
        branches (list): 要读取的分支列表。
        load_range (tuple, optional): 加载范围 (start, end)，范围在 [0, 1]。默认为 None。
        treename (str, optional): 树名称。默认为 None（自动检测）。
        branch_magic (dict, optional): 分支名称映射。默认为 None。

    Returns:
        ak.Array: 读取的数据。
    """
    import uproot

    with uproot.open(filepath) as f:
        if treename is None:
            treenames = {k.split(";")[0] for k, v in f.items() if getattr(v, "classname", "") == "TTree"}
            if len(treenames) == 1:
                treename = treenames.pop()
            else:
                raise RuntimeError(f"需要指定 `treename`，因为文件 {filepath} 中找到多个树: {str(treenames)}")
        tree = f[treename]
        if load_range is not None:
            start = math.trunc(load_range[0] * tree.num_entries)
            stop = max(start + 1, math.trunc(load_range[1] * tree.num_entries))
        else:
            start, stop = None, None
        if branch_magic is not None:
            branch_dict = {}
            for name in branches:
                decoded_name = name
                for src, tgt in branch_magic.items():
                    if src in decoded_name:
                        decoded_name = decoded_name.replace(src, tgt)
                branch_dict[name] = decoded_name
            outputs = tree.arrays(filter_name=list(branch_dict.values()), entry_start=start, entry_stop=stop)
            for name, decoded_name in branch_dict.items():
                if name != decoded_name:
                    outputs[name] = outputs[decoded_name]
        else:
            outputs = tree.arrays(filter_name=branches, entry_start=start, entry_stop=stop)
    return outputs


def _read_awkd(filepath, branches, load_range=None):
    """读取 Awkward 文件。

    Args:
        filepath (str): 文件路径。
        branches (list): 要读取的分支列表。
        load_range (tuple, optional): 加载范围 (start, end)，范围在 [0, 1]。默认为 None。

    Returns:
        ak.Array: 读取的数据。
    """
    try:
        import awkward0

        with awkward0.load(filepath) as f:
            outputs = {k: f[k] for k in branches}
    except ImportError:
        # 如果没有 awkward0，尝试直接使用 awkward
        outputs = ak.from_parquet(filepath, columns=branches)
        if isinstance(outputs, ak.Array) and hasattr(outputs, "fields"):
            outputs = {k: outputs[k] for k in branches}
        else:
            raise RuntimeError(f"无法读取文件 {filepath}，请检查格式")

    if load_range is None:
        load_range = (0, 1)
    start = math.trunc(load_range[0] * len(outputs[branches[0]]))
    stop = max(start + 1, math.trunc(load_range[1] * len(outputs[branches[0]])))
    for k, v in outputs.items():
        if isinstance(v, ak.Array):
            outputs[k] = v[start:stop]
        else:
            try:
                import awkward0

                outputs[k] = ak.from_awkward0(v[start:stop])
            except (ImportError, AttributeError):
                outputs[k] = v[start:stop]
    return ak.Array(outputs)


def _read_parquet(filepath, branches, load_range=None):
    """读取 Parquet 文件。

    Args:
        filepath (str): 文件路径。
        branches (list): 要读取的分支列表。
        load_range (tuple, optional): 加载范围 (start, end)，范围在 [0, 1]。默认为 None。

    Returns:
        ak.Array: 读取的数据。
    """
    outputs = ak.from_parquet(filepath, columns=branches)
    if load_range is not None:
        start = math.trunc(load_range[0] * len(outputs))
        stop = max(start + 1, math.trunc(load_range[1] * len(outputs)))
        outputs = outputs[start:stop]
    return outputs


def read_files(filelist, branches, load_range=None, show_progressbar=False, file_magic=None, **kwargs):
    """读取文件列表。

    Args:
        filelist: 文件路径列表或 glob 模式。
        branches (list): 要读取的分支列表。
        load_range (tuple, optional): 加载范围 (start, end)，范围在 [0, 1]。默认为 None。
        show_progressbar (bool): 是否显示进度条。默认为 False。
        file_magic (dict, optional): 文件魔法变量（根据文件名模式设置变量值）。默认为 None。
        **kwargs: 其他参数（如 treename, branch_magic）。

    Returns:
        ak.Array: 合并后的数据。
    """
    import glob
    import os
    import re

    if isinstance(filelist, str):
        filelist = glob.glob(filelist)
    elif not isinstance(filelist, (list, tuple)):
        filelist = list(filelist)

    branches = list(branches)
    table = []
    if show_progressbar:
        filelist = tqdm.tqdm(filelist)

    for filepath in filelist:
        ext = os.path.splitext(filepath)[1]
        if ext not in (".h5", ".root", ".awkd", ".parquet"):
            raise RuntimeError(f"不支持文件类型 `{ext}`：{filepath}")
        try:
            if ext == ".h5":
                a = _read_hdf5(filepath, branches, load_range=load_range)
            elif ext == ".root":
                a = _read_root(
                    filepath,
                    branches,
                    load_range=load_range,
                    treename=kwargs.get("treename", None),
                    branch_magic=kwargs.get("branch_magic", None),
                )
            elif ext == ".awkd":
                a = _read_awkd(filepath, branches, load_range=load_range)
            elif ext == ".parquet":
                a = _read_parquet(filepath, branches, load_range=load_range)
        except Exception:
            a = None
            _logger.error("读取文件 %s 时出错:", filepath)
            _logger.error(traceback.format_exc())

        if a is not None:
            if file_magic is not None:
                for var, value_dict in file_magic.items():
                    if var in a.fields:
                        warn_n_times(f"变量 `{var}` 已在数组中定义，但将被 file_magic {value_dict} 覆盖。")
                    a[var] = 0
                    for fn_pattern, value in value_dict.items():
                        if re.search(fn_pattern, filepath):
                            a[var] = value
                            break
            table.append(a)

    if len(table) == 0:
        raise RuntimeError(f"从文件列表 {filelist} 加载了零条记录，load_range={load_range}。")

    return _concat(table)
