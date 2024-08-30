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

====================================================================================================================================
PUQ: Python package for generating experimental designs tailored for uncertainty quantification, featuring parallel implementations
====================================================================================================================================

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

1) Extract the zipped file.

2) From the command line, go to the directory of the source code.

3) Use the following command to create a virtual environment::

  python3 -m venv venv/  
  source venv/bin/activate  
 
We note that creating a virtual environment is not a required step. However, we tested this
procedure to prevent any conflict, and the code runs smoothly.

Installation
~~~~~~~~~~~~

To install the PUQ package:

1) Go to the directory of the source code (if a user has not done so yet).

2) Use the following command to install the required packages::

 pip install -r requirements.txt

3) From the command line, use the following command to install PUQ::

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

To compile the documentation, first ensure that Sphinx is installed. Then, to
generate documentation, run command ``make html`` from terminal within this directory as follows ::

 cd docs
 make html

The HTML files are then stored in ``docs/_build/html``


**Citation:**

- Please use the following to cite PUQ in a publication:

.. code-block:: bibtex

   @techreport{PUQ2024,
     author      = {Özge Sürer, Matthew Plumlee, Stefan M. Wild},
     title       = {PUQ Users Manual},
     institution = {},
     number      = {Version 0.1.1},
     year        = {2024},
     url         = {https://github.com/parallelUQ/PUQ}
   }

Examples
~~~~~~~~

We provide examples in the ``examples/`` directory to illustrate the basic usage of PUQ. 
These examples replicate the results presented in the paper titled 'Performance Analysis of 
Sequential Experimental Design for Calibration in Parallel Computing Environments' 
by Sürer and Wild (2024).

**Instructions for running the illustrative examples with performance model**

To replicate Figures~6--9, respectively:

1)Go to the ``examples/`` directory.

2)Execute the followings from the command line::

 python3 Fig6.py
 python3 Fig7.py
 python3 Fig8.py
 python3 Fig9.py
 
Running each script should not take more than 90 sec. See the figures (jpeg files) saved under ``examples/`` directory.

**Instructions for running the sequential design using different synthetic simulation models**

To replicate Figure~2, execute the following from the command line::

 python3 Fig2.py
 
Running this script takes about a minute on a personal Mac laptop. 

To collect ``-max_eval`` simulation outputs from parameters acquired with 
different acquisition functions (``-al_func``) and synthetic simulation models (``-funcname``), 
one can use the script ``run_test_funcs.py``.

As an example, execute the followings from the command line::

 python3 run_test_funcs.py -funcname ackley -al_func ei -max_eval 200
 python3 run_test_funcs.py -funcname easom -al_func ei -max_eval 200
 python3 run_test_funcs.py -funcname himmelblau -al_func eivar -max_eval 200
 python3 run_test_funcs.py -funcname holder -al_func eivar -max_eval 200
 python3 run_test_funcs.py -funcname matyas -al_func hybrid_ei -max_eval 200
 python3 run_test_funcs.py -funcname sphere -al_func hybrid_ei -max_eval 200
 
Once completed, ``Figure_funcname.jpg`` is saved under ``examples/`` directory.
Running each script should not take more than one minute on a personal Mac laptop.
  
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
