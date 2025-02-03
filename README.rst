|

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT
    :alt: License

.. image:: https://github.com/parallelUQ/PUQ/actions/workflows/puq-ci.yml/badge.svg?/branch=main
    :target: https://github.com/parallelUQ/PUQ/actions

.. image:: https://readthedocs.org/projects/puq/badge/?version=latest
    :target: https://puq.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

|

.. after_badges_rst_tag

======================================================================================================================================
PUQ: Python package for generating experimental designs tailored for uncertainty quantification and featuring parallel implementations
======================================================================================================================================

Dependencies
~~~~~~~~~~~~

PUQ is a Python package that employs novel experimental design techniques with intelligent selection criteria, 
refining data collection to enhance the efficiency and effectiveness of uncertainty quantification.

This code is tested with Python 3.9, 3.10, and 3.11 and requires pip.

In the following, we provide the instructions to replicate examples from the paper titled 'Batch sequential
experimental design for calibration of stochastic simulation models.'

This code is publicly available at `PUQ Github Repo <https://github.com/parallelUQ/PUQ/tree/dev/batch_sequential?tab=readme-ov-file>`_.

Set up 
~~~~~~

We recommend creating a Python virtual environment within the working directory of PUQ. 
If a virtual environment is created, PUQ's required packages are installed and 
isolated from those installed a priori. Creating a virtual environment will also prevent
having conflicting packages on a user's machine. You may need to install the virtual 
environment on your system (if a user's system does not have it), for example, 
with 'apt install python3.9-venv'

1) Extract the zipped file.

2) From the command line, go to the directory of the source code.

3) Use the following command to create a virtual environment::

    python -m venv venv/  
    source venv/bin/activate  
 
We note that creating a virtual environment is not a required step. However, we tested this
procedure to prevent any conflict, and the code runs smoothly.

Installation
~~~~~~~~~~~~

To install the PUQ package:

1) Navigate to the source code directory where ``setup.py`` is located

2) Use the following commands to ensure that the environment has the latest tools for package management and distribution::

    python -m ensurepip --upgrade
    python -m pip install --upgrade setuptools
        
3) Use the following command to install the required packages::

    pip install -r requirements.txt
    
4) Use to following command to install the wheel package::

    pip install --upgrade setuptools wheel 

5) From the command line, use the following command to install PUQ::

    pip install -e .

Once installed, a user should see ``build/`` directory created.

 
Examples
~~~~~~~~

We provide all the examples presented in the paper in the `examples </examples>`_. 

Instructions for running the illustrative examples:

To replicate Figures~1--3, respectively:

1) Go to the examples/ directory.

2) Execute the followings from the command line:

    python Figure1_2.py
    python Figure3.py

Running each script should not take more than 60 sec. See the figures (png files) saved under examples/ directory.




Running each script should not take more than 60 sec. See the figures (png files) saved under ``examples/`` directory.

  
Final comments
~~~~~~~~~~~~~~

Type ``deactivate`` from the command line to deactivate the virtual environment if created.

Type ``pip uninstall PUQ`` from the command line to uninstall the package.



**Citation:**

- Please use the following to cite PUQ in a publication:

.. code-block:: bibtex

   @techreport{PUQ2022,
     author      = {Özge Sürer, Matthew Plumlee, Stefan M. Wild},
     title       = {PUQ Users Manual},
     institution = {},
     number      = {Version 0.1.0},
     year        = {2022},
     url         = {https://github.com/parallelUQ/PUQ}
   }



.. _NumPy: http://www.numpy.org
.. _pytest-cov: https://pypi.org/project/pytest-cov/
.. _pytest: https://pypi.org/project/pytest/
.. _Python: http://www.python.org
.. _SciPy: http://www.scipy.org
.. _libEnsemble: https://libensemble.readthedocs.io/en/main/
