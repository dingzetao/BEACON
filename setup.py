from setuptools import find_packages, setup

setup(
    name="beacon-pathology",
    version="0.1.0",
    description="BEACON: Attention-based cross-modal decoding of spatial onco-niches from H&E",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torch-geometric>=2.4.0",
        "numpy>=1.23",
        "pandas>=1.5",
        "scipy>=1.9",
        "scikit-learn>=1.2",
        "matplotlib>=3.6",
        "opencv-python>=4.7",
        "tqdm>=4.64",
        "PyYAML>=6.0",
    ],
)
