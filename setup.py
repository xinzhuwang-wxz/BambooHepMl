"""
BambooHepMl 安装配置
"""

from pathlib import Path

from setuptools import find_packages, setup

# 读取 README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# 读取 requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = requirements_file.read_text(encoding="utf-8").strip().split("\n")
    requirements = [r.strip() for r in requirements if r.strip() and not r.startswith("#")]

setup(
    name="bamboohepml",
    version="0.1.0",
    description="高能物理机器学习框架",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="BambooHepMl Contributors",
    url="https://github.com/BambooHepMl/BambooHepMl",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-xdist>=3.0.0",
            "fastapi[all]>=0.100.0",
            "httpx>=0.24.0",
        ],
        "serve": [
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
            "ray[serve]>=2.5.0",
            "onnxruntime>=1.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bamboohepml=bamboohepml.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
