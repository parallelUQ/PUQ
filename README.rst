==================================================================================
PUQ: Python package for parallel uncertainty quantification
==================================================================================


Dependencies
~~~~~~~~~~~~

PUQ is a Python package for novel uncertainty quantification techniques, 
specifically to be able to run in a parallel environment.

This code requires Python (version 3.6 or later) and pip. 

Examples under ``examples/`` directory replicate figures 1--5.

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

As an alternative to 'pip install -e .' in Step 3, the package can be installed from the .tar file as well::

 python3 setup.py sdist bdist_wheel 
 pip install ./dist/PUQ-0.1.0.tar.gz
 

Instructions for running the illustrative examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To replicate Figures~1--4, respectively:

1)Go to the ``examples/`` directory.

2)Execute the followings from the command line::

 python3 Figure1.py
 python3 Figure2a.py
 python3 Figure2b.py
 python3 Figure3.py
 python3 Figure4.py

Running each script should not take more than 60 sec. See the figures (png files) saved under ``examples/`` directory.

Instructions for running one of the prominent empirical results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instructions are provided to replicate the first two panels in Figure~6.

To replicate, execute the following from the command line::

 python3 Figure5ab.py 

Running this script takes about 3hrs on a personal Mac laptop. 
Once completed, ``Figure5a.png`` and ``Figure5b.png`` are saved under ``examples/`` directory.
  
Final comments
~~~~~~~~~~~~~~

Type ``deactivate`` from the command line to deactivate the virtual environment if created.

Type ``pip uninstall PUQ`` from the command line to uninstall the package.

