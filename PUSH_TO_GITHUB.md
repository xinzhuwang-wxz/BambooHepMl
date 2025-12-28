# 推送到 GitHub 的步骤

## 1. 检查当前状态

```bash
cd /Users/physicsboy/Documents/GitHub/BambooHepMl
git status
```

## 2. 配置远程仓库（如果还没有）

```bash
# 如果 remote 指向错误的仓库，先删除
git remote remove origin

# 添加正确的 remote
git remote add origin https://github.com/xinzhuwang-wxz/BambooHepMl.git

# 验证
git remote -v
```

## 3. 添加所有文件并提交

```bash
# 添加所有文件
git add -A

# 查看将要提交的文件
git status

# 提交
git commit -m "Initial commit: BambooHepMl framework

- Data and feature system (weaver-core inspired)
- ML pipeline (Made-With-ML inspired)
- Models, engine, tasks, scheduler, serve modules
- CLI system with train/predict/export/inspect commands
- FastAPI and Ray Serve integration
- ONNX inference support
- MLflow and TensorBoard integration
- pytest test structure
- GitHub Actions CI/CD
- MkDocs documentation system"
```

## 4. 推送到 GitHub

```bash
# 设置主分支为 main（如果还没有）
git branch -M main

# 推送到 GitHub
git push -u origin main
```

如果遇到权限问题，可能需要：
- 配置 GitHub 认证（使用 Personal Access Token 或 SSH key）
- 或者使用 GitHub CLI: `gh auth login`

## 5. 启用 GitHub Pages

推送完成后：

1. 访问 GitHub 仓库：https://github.com/xinzhuwang-wxz/BambooHepMl
2. 进入 Settings → Pages
3. 在 Source 部分选择 "GitHub Actions"
4. 保存设置

文档将自动通过 GitHub Actions 构建并部署到 GitHub Pages。

## 6. 查看文档

文档部署完成后（通常需要几分钟），可以通过以下地址访问：

```
https://xinzhuwang-wxz.github.io/BambooHepMl/
```

## 7. 本地预览文档（可选）

在推送之前，可以本地预览文档：

```bash
# 安装依赖
pip install mkdocs mkdocstrings[python] griffe

# 启动本地服务器
mkdocs serve

# 浏览器访问 http://127.0.0.1:8000
```

## 注意事项

- 确保 GitHub 仓库已创建（如果还没有，在 GitHub 上创建）
- 如果使用 HTTPS，可能需要配置 Personal Access Token
- 如果使用 SSH，确保 SSH key 已添加到 GitHub
- GitHub Actions 需要仓库设置为 public 或启用 GitHub Actions 权限

