|

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT
    :alt: License

.. image:: https://github.com/parallelUQ/PUQ/actions/workflows/puq-ci.yml/badge.svg?/branch=main
    :target: https://github.com/parallelUQ/PUQ/actions

.. image:: https://readthedocs.org/projects/puq/badge/?version=latest
    :target: https://puq.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

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

Set up 
~~~~~~

We recommend creating a Python virtual environment within the working directory of PUQ. 
If a virtual environment is created, PUQ's required packages are installed and 
isolated from those installed a priori. Creating a virtual environment will also prevent
having conflicting packages on a user's machine. You may need to install the virtual 
environment on your system (if a user's system does not have it), for example, 
with 'apt install python3.9-venv'

1)Extract the zipped file.

2)From the command line, go to the directory of the source code.

3)Use the following command to create a virtual environment::

  python3 -m venv venv/  
  source venv/bin/activate  
 
We note that creating a virtual environment is not a required step. However, we tested this
procedure to prevent any conflict, and the code runs smoothly.

Installation
~~~~~~~~~~~~

To install the PUQ package:

1)Go to the directory of the source code (if a user has not done so yet).

2)Use the following command to install the required packages::

 pip install -r requirements.txt

3)From the command line, use the following command to install PUQ::

 pip install -e .

Once installed, a user should see ``build/`` directory created.

 
Testing
~~~~~~~

The test suite requires the pytest_ and pytest-cov_ packages to be installed
and can be run from the ``tests/`` directory of the source distribution by running::

./run-tests.sh


Documentation
~~~~~~~~~~~~~

The documentation is stored in ``docs/`` and is compiled with the Sphinx Python
documentation generator. It is written in the reStructuredText format. These
files are hosted at `Read the Docs <http://PUQ.readthedocs.io>`_.

To compile the documentation, from the directory of the source code, run the following command :: 

 sphinx-build -M html docs docs

The HTML files are then stored in ``docs/html``


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

Examples
~~~~~~~~

We provide examples in the ``examples/`` directory to illustrate the basic usage of PUQ. 
These examples replicate the results presented in the paper titled 'Sequential Bayesian 
Experimental Design for Calibration of Expensive Simulation Models' by Sürer, Plumlee, and Wild (2024).

**Instructions for running the illustrative examples**

To replicate Figures~1--3, respectively:

1)Go to the ``examples/`` directory.

2)Execute the followings from the command line::

 python3 figure1b.py
 python3 figure2.py
 python3 figure3.py

Running each script should not take more than 60 sec. See the figures (png files) saved under ``examples/`` directory.

**Instructions for running the prominent empirical results**

Instructions are provided to replicate each panel in Figure~6.

To replicate the upper-left panel (banana function), execute the following from the command line::

 python3 figure6.py -funcname banana
 
Running this script takes about 2.5hrs on a personal Mac laptop. 
Once completed, ``Figure6_banana.png`` is saved under ``examples/`` directory.
 
To replicate the upper-right panel (bimodal function), execute the following from the command line::

 python3 figure6.py -funcname bimodal

Running this script takes about 2.5hrs on a personal Mac laptop. 
Once completed, ``Figure6_bimodal.png`` is saved under ``examples/`` directory.
 
To replicate the lower-left panel (unimodal function), execute the following from the command line::

 python3 figure6.py -funcname unimodal

Running this script takes about 2hr on a personal Mac laptop. 
Once completed, ``Figure6_unimodal.png`` is saved under ``examples/`` directory.
 
To replicate the lower-right panel (unidentifiable function), execute the following from the command line::

 python3 figure6.py -funcname unidentifiable
 
Running this script takes about 2.5hrs on a personal Mac laptop. 
Once completed, ``Figure6_unidentifiable.png`` is saved under ``examples/`` directory.
  
Final comments
~~~~~~~~~~~~~~

Type ``deactivate`` from the command line to deactivate the virtual environment if created.

Type ``pip uninstall PUQ`` from the command line to uninstall the package.


.. _NumPy: http://www.numpy.org
.. _pytest-cov: https://pypi.org/project/pytest-cov/
.. _pytest: https://pypi.org/project/pytest/
.. _Python: http://www.python.org
.. _SciPy: http://www.scipy.org
.. _libEnsemble: https://libensemble.readthedocs.io/en/main/
