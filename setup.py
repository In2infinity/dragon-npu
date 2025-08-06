#!/usr/bin/env python3
"""
DragonNPU Setup Script
Complete installation system for DragonNPU framework
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

setup(
    name="dragon-npu",
    version="1.0.0",
    author="DragonNPU Team",
    description="🐉 DragonNPU - Bringing AI acceleration to Linux",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dragonfire/dragon-npu",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "rich>=10.0.0",
        "click>=8.0.0",
        "psutil>=5.8.0",
        "pyyaml>=5.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.900",
        ],
        "ml": [
            "torch>=1.9.0",
            "onnx>=1.10.0",
            "onnxruntime>=1.8.0",
            "tensorflow>=2.6.0",
            "transformers>=4.20.0",
        ],
        "vision": [
            "opencv-python>=4.5.0",
            "pillow>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dragon-npu=dragon_npu_cli:main",
            "dnpu=dragon_npu_cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "dragon_npu": [
            "backends/*.py",
            "examples/*.py",
            "configs/*.yaml",
        ],
    },
)