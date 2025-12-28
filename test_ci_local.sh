#!/bin/bash
# 本地 CI 测试脚本
# 模拟 GitHub Actions CI 流程

set -e  # 遇到错误立即退出

echo "=========================================="
echo "BambooHepMl 本地 CI 测试"
echo "=========================================="
echo ""

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查函数
check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✓${NC} $1 已安装"
        return 0
    else
        echo -e "${RED}✗${NC} $1 未安装"
        return 1
    fi
}

# 1. 检查必要的命令
echo "1. 检查必要的命令..."
check_command python3 || exit 1
check_command pip || exit 1
echo ""

# 2. Lint 检查
echo "=========================================="
echo "2. 运行 Lint 检查"
echo "=========================================="

# 检查 flake8
if ! check_command flake8; then
    echo "安装 flake8..."
    pip install flake8
fi

# 检查 black
if ! check_command black; then
    echo "安装 black..."
    pip install black
fi

# 检查 isort
if ! check_command isort; then
    echo "安装 isort..."
    pip install isort
fi

echo ""
echo "运行 flake8..."
flake8 bamboohepml --count --select=E9,F63,F7,F82 --show-source --statistics || {
    echo -e "${RED}✗${NC} flake8 检查失败"
    exit 1
}
flake8 bamboohepml --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics || {
    echo -e "${YELLOW}⚠${NC} flake8 有警告（非致命）"
}

echo ""
echo "运行 black 检查..."
black --check bamboohepml tests || {
    echo -e "${RED}✗${NC} black 检查失败"
    echo "运行 'make style' 或 'black bamboohepml tests' 来修复"
    exit 1
}

echo ""
echo "运行 isort 检查..."
isort --check-only bamboohepml tests || {
    echo -e "${RED}✗${NC} isort 检查失败"
    echo "运行 'make style' 或 'isort bamboohepml tests' 来修复"
    exit 1
}

echo -e "${GREEN}✓${NC} Lint 检查通过"
echo ""

# 3. 安装依赖
echo "=========================================="
echo "3. 安装依赖"
echo "=========================================="

echo "升级 pip..."
python3 -m pip install --upgrade pip

echo "安装项目依赖..."
pip install -e . || {
    echo -e "${RED}✗${NC} 安装基础依赖失败"
    exit 1
}

echo "安装测试依赖..."
pip install -e ".[test]" || {
    echo -e "${RED}✗${NC} 安装测试依赖失败"
    exit 1
}

echo -e "${GREEN}✓${NC} 依赖安装完成"
echo ""

# 4. 运行测试
echo "=========================================="
echo "4. 运行单元测试"
echo "=========================================="

# 检查 pytest
if ! check_command pytest; then
    echo "安装 pytest..."
    pip install pytest pytest-cov pytest-xdist
fi

echo "运行 pytest..."
pytest tests/ -v --cov=bamboohepml --cov-report=term --cov-report=html -m "not slow and not integration" || {
    echo -e "${RED}✗${NC} 测试失败"
    exit 1
}

echo -e "${GREEN}✓${NC} 单元测试通过"
echo ""

# 5. 构建文档
echo "=========================================="
echo "5. 构建文档"
echo "=========================================="

# 检查 mkdocs
if ! check_command mkdocs; then
    echo "安装 mkdocs..."
    pip install 'mkdocs>=1.4.0' 'mkdocstrings[python]>=0.18.0' 'griffe<0.40'
fi

echo "构建文档..."
mkdocs build || {
    echo -e "${RED}✗${NC} 文档构建失败"
    exit 1
}

echo -e "${GREEN}✓${NC} 文档构建成功"
echo ""

# 6. 总结
echo "=========================================="
echo -e "${GREEN}✓ 所有检查通过！${NC}"
echo "=========================================="
echo ""
echo "可以安全地推送到 GitHub："
echo "  git add ."
echo "  git commit -m 'Your commit message'"
echo "  git push origin main"
echo ""
