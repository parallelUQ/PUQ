import setuptools
from setuptools import setup, Extension
from setuptools.command.test import test as TestCommand
import numpy

class Run_TestSuite(TestCommand):
    def run_tests(self):
        import os
        import sys
        py_version = sys.version_info[0]
        print('Python version from setup.py is', py_version)
        run_string = "tests/run-tests.sh -p " + str(py_version)
        os.system(run_string)

with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="PUQ",
    version="0.1.1",
    author="Özge Sürer, Matthew Plumlee, Stefan M. Wild",
    author_email="surero@miamioh.edu",
    description="Uncertainty quantification methods with parallel implementations",
    url="https://github.com/parallelUQ/PUQ",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    setup_requires=[
        'setuptools>=18.0',
        'cython'
    ],
    install_requires=[
                      'numpy',
                      'scipy',
                      'libensemble',
                      'torch',
                      'scikit-learn',
                      'smt',
                      'numpy'
                      ],
    tests_require=["pytest", "pytest-cov", "flake8"],
    extras_require={'docs': ['sphinx', 'sphinxcontrib.bibtex', 'sphinx_rtd_theme']},
    cmdclass={'test': Run_TestSuite},
    ext_modules=[
        Extension('PUQ.surrogatesupport.matern_covmat', sources=['PUQ/surrogatesupport/matern_covmat.pyx']),
    ],
    include_dirs=[numpy.get_include()]
)
