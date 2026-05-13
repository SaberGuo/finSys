from setuptools import setup, find_packages

setup(
    name="kline-vit",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "mplfinance>=0.12.10b0",
        "backtrader>=1.9.78.123",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "Pillow>=10.0.0",
        "pyyaml>=6.0",
        "click>=8.1.0",
    ],
    entry_points={
        "console_scripts": [
            "kline-vit=kline_vit.cli:main",
        ],
    },
    python_requires=">=3.10",
)
