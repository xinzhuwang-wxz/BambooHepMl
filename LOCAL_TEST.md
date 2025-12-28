# 本地 CI 测试指南

在推送到 GitHub 之前，请在本地运行以下测试以确保 CI 能够通过。

## 快速测试

运行测试脚本（推荐）：

```bash
cd /Users/physicsboy/Documents/GitHub/BambooHepMl
bash test_ci_local.sh
```

## 手动测试步骤

如果脚本无法运行，可以手动执行以下步骤：

### 1. Lint 检查

```bash
# 安装依赖（如果还没有）
pip install flake8 black isort

# 运行 flake8
flake8 bamboohepml --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 bamboohepml --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# 运行 black 检查
black --check bamboohepml tests

# 运行 isort 检查
isort --check-only bamboohepml tests
```

如果 black 或 isort 检查失败，运行以下命令自动修复：

```bash
make style
# 或者
black bamboohepml tests
isort bamboohepml tests
```

### 2. 安装项目依赖

```bash
# 升级 pip
python3 -m pip install --upgrade pip

# 安装基础依赖
pip install -e .

# 安装测试依赖
pip install -e ".[test]"
```

### 3. 运行单元测试

```bash
# 安装 pytest（如果还没有）
pip install pytest pytest-cov pytest-xdist

# 运行测试
pytest tests/ -v --cov=bamboohepml --cov-report=term -m "not slow and not integration"
```

### 4. 构建文档

```bash
# 安装 mkdocs（如果还没有）
pip install 'mkdocs>=1.4.0' 'mkdocstrings[python]>=0.18.0' 'griffe<0.40'

# 构建文档
mkdocs build
```

### 5. 本地预览文档（可选）

```bash
mkdocs serve
# 然后在浏览器中访问 http://127.0.0.1:8000
```

## 使用 Makefile（更简单）

如果已经安装了所有依赖，可以使用 Makefile：

```bash
# 代码风格检查
make lint

# 运行测试
make test

# 构建文档
make docs

# 或者一次性运行所有检查
make lint && make test && make docs
```

## 常见问题

### 1. 权限错误

如果遇到权限错误，尝试：

```bash
# 使用用户安装
pip install --user <package>

# 或者使用虚拟环境
python3 -m venv venv
source venv/bin/activate
pip install -e ".[test]"
```

### 2. 导入错误

确保已安装所有依赖：

```bash
pip install -e .
pip install -e ".[test]"
```

### 3. 测试失败

检查测试输出中的具体错误信息，通常是因为：
- 缺少依赖（检查 `requirements.txt`）
- 测试数据文件不存在（某些测试可能需要数据文件）
- 环境配置问题

## 检查清单

在推送之前，确保：

- [ ] Lint 检查通过（flake8, black, isort）
- [ ] 所有依赖正确安装
- [ ] 单元测试通过
- [ ] 文档能够成功构建
- [ ] 没有明显的错误或警告

## 推送命令

所有检查通过后：

```bash
git add .
git commit -m "Fix CI: Update GitHub Actions and add missing dependencies"
git push origin main
```

