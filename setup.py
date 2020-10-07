from setuptools import setup, find_packages

setup(
    name="nnkek",
    version="0.1.0",
    packages=find_packages(exclude=["*.tests", "*.cache", "tmp", "*.tests.*", "tests.*", "tests"]),
    license="unlicense",
    install_requires=[
        "tqdm",
        "numpy",
        "pillow",
        "opencv",
        "matplotlib",
        "tools",
        "albumentations",
        "oss2",
        "torch",
        "seaborn",
        "tensorflow",
        "torchvision",
        "scikit",
        "pandas",
        "setuptools",
        "requests",
        "docopt",
    ],
)
