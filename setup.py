import setuptools
from setuptools import setup
import numpy


setup(
    name="PUQ",
    version="0.1.0",
    author="Özge Sürer, Matthew Plumlee, Stefan M. Wild",
    author_email="surero@miamioh.edu",
    description="Python package for generating experimental designs tailored for uncertainty quantification, featuring parallel implementations",
    url="https://github.com/parallelUQ/PUQ",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    include_dirs=[numpy.get_include()],
)
