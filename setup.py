import setuptools
from setuptools import setup, Extension
import numpy


setup(
    name="PUQ",
    version="0.1.0",
    author="Blinded Authors",
    description="Uncertainty quantification methods with parallel implementations",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "pandas",
        "matplotlib",
        "smt",
    ],
    include_dirs=[numpy.get_include()],
)