|

.. image:: https://github.com/parallelUQ/PUQ/actions/workflows/puq-ci.yml/badge.svg?/branch=main
    :target: https://github.com/parallelUQ/PUQ/actions

|
    
==================================================================================
PUQ: Python package for parallel uncertainty quantification
==================================================================================


Dependencies
~~~~~~~~~~~~

PUQ has been tested on Unix/Linux and MacOS systems.

PUQ's base has the following dependencies:

 * Python_ 3.6+
 * NumPy_ -- for data structures and performant numerical linear algebra
 * SciPy_ -- for scientific calculations needed for specific modules
 * libEnsemble_ -- for parallel implementation


Installation
~~~~~~~~~~~~

From the command line, use the following command to install PUQ::

 pip install PUQ==0.1.0


Alternatively, the source code can be downloaded to the local folder, and the
package can be installed from the .tar file.

Testing
~~~~~~~

The test suite requires the pytest_ and pytest-cov_ packages to be installed
and can be run from the ``tests/`` directory of the source distribution by running::

./run-tests.sh

If you have the source distribution, you can run the tests in the top-level
directory containing the setup script with ::

 python setup.py test

Further options are available for testing. To see a complete list of options, run::

 ./run-tests.sh -h

Coverage reports are produced under the relevant directory only if all tests are used.

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

   @techreport{PUQ2022,
     author      = {Özge Sürer},
     title       = {PUQ Users Manual},
     institution = {Miami University},
     number      = {Version 0.1.0},
     year        = {2021},
     url         = {https://PUQ.readthedocs.io}
   }

Examples
~~~~~~~~

We provide examples in the ``examples/`` directory to illustrate the basic usage
of PUQ.

.. _NumPy: http://www.numpy.org
.. _pytest-cov: https://pypi.org/project/pytest-cov/
.. _pytest: https://pypi.org/project/pytest/
.. _Python: http://www.python.org
.. _SciPy: http://www.scipy.org
.. _libEnsemble: https://libensemble.readthedocs.io/en/main/
