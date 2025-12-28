"""
表达式工具模块

借鉴 weaver-core 的 expr 表达式系统，支持：
- 从表达式提取变量名
- 表达式求值
- HEP 常用函数（pad, clip, p4 等）
"""
import math
import numpy as np
import awkward as ak


def _hash(*args):
    """计算多个数组的哈希值。

    Args:
        *args: 输入数组。

    Returns:
        np.ndarray: 哈希值数组。
    """
    return np.array([x.__hash__() for x in zip(*args)])


def _concat(arrays, axis=0):
    """连接数组。

    Args:
        arrays: 要连接的数组列表。
        axis (int): 连接轴。默认为 0。

    Returns:
        连接后的数组。
    """
    if len(arrays) == 0:
        return np.array([])
    if isinstance(arrays[0], np.ndarray):
        return np.concatenate(arrays, axis=axis)
    else:
        return ak.concatenate(arrays, axis=axis)


def _stack(arrays, axis=1):
    """堆叠数组。

    Args:
        arrays: 要堆叠的数组列表。
        axis (int): 堆叠轴。默认为 1。

    Returns:
        堆叠后的数组。
    """
    if len(arrays) == 0:
        return np.array([])
    if isinstance(arrays[0], np.ndarray):
        return np.stack(arrays, axis=axis)
    else:
        s = [slice(None)] * (arrays[0].ndim + 1)
        s[axis] = np.newaxis
        s = tuple(s)
        return ak.concatenate([a.__getitem__(s) for a in arrays], axis=axis)


def _pad(a, maxlen, value=0, dtype='float32'):
    """将数组填充到指定长度。

    Args:
        a: 输入数组（可以是 numpy 或 awkward 数组）。
        maxlen (int): 目标长度。
        value: 填充值。默认为 0。
        dtype: 数据类型。默认为 'float32'。

    Returns:
        填充后的数组。
    """
    if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
        return a
    elif isinstance(a, ak.Array):
        if a.ndim == 1:
            a = ak.unflatten(a, 1)
        a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
        return ak.values_astype(a, dtype)
    else:
        x = (np.ones((len(a), maxlen)) * value).astype(dtype)
        for idx, s in enumerate(a):
            if not len(s):
                continue
            trunc = s[:maxlen].astype(dtype)
            x[idx, :len(trunc)] = trunc
        return x


def _repeat_pad(a, maxlen, shuffle=False, dtype='float32'):
    """重复填充数组到指定长度（每个事件独立处理）。

    Args:
        a: 输入数组（awkward Array）。
        maxlen (int): 目标长度。
        shuffle (bool): 是否打乱。默认为 False。
        dtype: 数据类型。默认为 'float32'。

    Returns:
        填充后的数组。
    """
    if isinstance(a, ak.Array):
        # 对每个事件分别处理
        padded_list = []
        for i in range(len(a)):
            event_data = a[i]
            if len(event_data) == 0:
                # 空事件，用 0 填充
                padded = np.zeros(maxlen, dtype=dtype)
            elif len(event_data) >= maxlen:
                # 长度足够，截断
                padded = ak.to_numpy(event_data[:maxlen]).astype(dtype)
            else:
                # 需要填充：重复原始数据
                event_np = ak.to_numpy(event_data).astype(dtype)
                # 计算需要重复多少次
                n_repeats = int(np.ceil(maxlen / len(event_np)))
                repeated = np.tile(event_np, n_repeats)
                padded = repeated[:maxlen]
                if shuffle:
                    # 只打乱 padding 部分
                    original_len = len(event_np)
                    padding_part = padded[original_len:]
                    np.random.shuffle(padding_part)
                    padded[original_len:] = padding_part
            padded_list.append(padded)
        return np.array(padded_list)
    else:
        # numpy array 的情况（向后兼容）
        if isinstance(a, np.ndarray) and a.ndim == 1:
            a = a.reshape(-1, 1)
        result = np.zeros((len(a), maxlen), dtype=dtype)
        for i, event_data in enumerate(a):
            if len(event_data) == 0:
                continue
            elif len(event_data) >= maxlen:
                result[i] = event_data[:maxlen]
            else:
                event_np = np.array(event_data).astype(dtype)
                n_repeats = int(np.ceil(maxlen / len(event_np)))
                repeated = np.tile(event_np, n_repeats)
                result[i] = repeated[:maxlen]
                if shuffle:
                    original_len = len(event_np)
                    padding_part = result[i, original_len:]
                    np.random.shuffle(padding_part)
                    result[i, original_len:] = padding_part
        return result


def _clip(a, a_min, a_max):
    """裁剪数组值到指定范围。

    Args:
        a: 输入数组。
        a_min: 最小值。
        a_max: 最大值。

    Returns:
        裁剪后的数组。
    """
    if isinstance(a, np.ndarray) or a.ndim == 1:
        return np.clip(a, a_min, a_max)
    else:
        return ak.unflatten(np.clip(ak.to_numpy(ak.flatten(a)), a_min, a_max), ak.num(a))


def _knn(support, query, k, n_jobs=1):
    """K 近邻查找。

    Args:
        support: 支持点集。
        query: 查询点集。
        k (int): 近邻数量。
        n_jobs (int): 并行作业数。默认为 1。

    Returns:
        近邻索引。
    """
    from scipy.spatial import cKDTree
    kdtree = cKDTree(support)
    d, idx = kdtree.query(query, k, n_jobs=n_jobs)
    return idx


def _batch_knn(supports, queries, k, maxlen_s, maxlen_q=None, n_jobs=1):
    """批量 K 近邻查找。

    Args:
        supports: 支持点集列表。
        queries: 查询点集列表。
        k (int): 近邻数量。
        maxlen_s (int): 支持点集最大长度。
        maxlen_q (int, optional): 查询点集最大长度。默认为 None（使用 maxlen_s）。
        n_jobs (int): 并行作业数。默认为 1。

    Returns:
        批量近邻索引。
    """
    assert (len(supports) == len(queries))
    if maxlen_q is None:
        maxlen_q = maxlen_s
    batch_knn_idx = np.ones((len(supports), maxlen_q, k), dtype='int32') * (maxlen_s - 1)
    for i, (s, q) in enumerate(zip(supports, queries)):
        batch_knn_idx[i, :len(q[:maxlen_q]), :] = _knn(
            s[:maxlen_s], q[:maxlen_q], k, n_jobs=n_jobs).reshape((-1, k))
    return batch_knn_idx


def _batch_permute_indices(array):
    """批量随机排列索引。

    Args:
        array: 输入数组。

    Returns:
        排列后的索引。
    """
    random_array = ak.unflatten(np.random.rand(ak.count(array)), ak.num(array))
    return ak.argsort(random_array)


def _batch_argsort(array):
    """批量排序索引。

    Args:
        array: 输入数组。

    Returns:
        排序后的索引。
    """
    return ak.argsort(array)


def _batch_gather(array, indices):
    """批量收集元素。

    Args:
        array: 输入数组。
        indices: 索引数组。

    Returns:
        收集后的数组。
    """
    return array[indices]


def _p4_from_pxpypze(px, py, pz, energy):
    """从 px, py, pz, energy 构建四动量。

    Args:
        px: x 方向动量。
        py: y 方向动量。
        pz: z 方向动量。
        energy: 能量。

    Returns:
        四动量向量。
    """
    try:
        import vector
        vector.register_awkward()
        return vector.zip({'px': px, 'py': py, 'pz': pz, 'energy': energy})
    except ImportError:
        raise ImportError("需要安装 vector 包：pip install vector")


def _p4_from_ptetaphie(pt, eta, phi, energy):
    """从 pt, eta, phi, energy 构建四动量。

    Args:
        pt: 横向动量。
        eta: 赝快度。
        phi: 方位角。
        energy: 能量。

    Returns:
        四动量向量。
    """
    try:
        import vector
        vector.register_awkward()
        return vector.zip({'pt': pt, 'eta': eta, 'phi': phi, 'energy': energy})
    except ImportError:
        raise ImportError("需要安装 vector 包：pip install vector")


def _p4_from_ptetaphim(pt, eta, phi, mass):
    """从 pt, eta, phi, mass 构建四动量。

    Args:
        pt: 横向动量。
        eta: 赝快度。
        phi: 方位角。
        mass: 质量。

    Returns:
        四动量向量。
    """
    try:
        import vector
        vector.register_awkward()
        return vector.zip({'pt': pt, 'eta': eta, 'phi': phi, 'mass': mass})
    except ImportError:
        raise ImportError("需要安装 vector 包：pip install vector")


def _get_variable_names(expr, exclude=['awkward', 'ak', 'np', 'numpy', 'math', 'len']):
    """从表达式中提取变量名。

    Args:
        expr (str): 表达式字符串。
        exclude (list): 要排除的名称列表。

    Returns:
        list: 变量名列表（已排序）。
    """
    import ast
    root = ast.parse(expr)
    return sorted({node.id for node in ast.walk(root) if isinstance(
        node, ast.Name) and not node.id.startswith('_')} - set(exclude))


def _eval_expr(expr, table):
    """求值表达式。

    Args:
        expr (str): 表达式字符串。
        table: 包含变量的表（awkward Array 或字典）。

    Returns:
        表达式求值结果。
    """
    tmp = {k: table[k] for k in _get_variable_names(expr)}
    tmp.update({
        'math': math,
        'np': np,
        'numpy': np,
        'ak': ak,
        'awkward': ak,
        'len': len,
        '_hash': _hash,
        '_concat': _concat,
        '_stack': _stack,
        '_pad': _pad,
        '_repeat_pad': _repeat_pad,
        '_clip': _clip,
        '_batch_knn': _batch_knn,
        '_batch_permute_indices': _batch_permute_indices,
        '_batch_argsort': _batch_argsort,
        '_batch_gather': _batch_gather,
        '_p4_from_pxpypze': _p4_from_pxpypze,
        '_p4_from_ptetaphie': _p4_from_ptetaphie,
        '_p4_from_ptetaphim': _p4_from_ptetaphim,
    })
    return eval(expr, tmp)

