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

This code is tested with Python 3.9, 3.10, 3.11, and 3.12 and requires pip.

Python 3.12 users need to install setuptools manually via::

    python -m ensurepip --upgrade
    python -m pip install --upgrade setuptools

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

The test suite requires the pytest_ and pytest-cov_ packages, which can be installed via ``pip install pytest pytest-cov``.

The test suite can be run from the ``tests/`` directory of the source distribution by running::

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

We provide examples in the `examples </examples>`_
directory to illustrate the basic usage of PUQ. 

These examples replicate some results presented in our papers. 
`README` file within the directory provides further instructions.

  
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
