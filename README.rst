==================================================================================
PUQ: Python package for parallel uncertainty quantification
==================================================================================


Dependencies
~~~~~~~~~~~~

PUQ is a Python package for the method introduced in the paper titled 'Sequential
Bayesian experimental design for calibration of expensive simulation models.'

This code requires Python (version 3.6 or later) and pip. 

Examples under ``examples/`` directory replicate figures 1--3 and figure 6.

Set up 
~~~~~~

We recommend creating a Python virtual environment within the working directory of PUQ. 
If a virtual environment is created, PUQ's required packages are installed and 
isolated from those installed a priori. Creating a virtual environment will also prevent
having conflicting packages on a user's machine. 

1)Extract the zipped file.

2)From the command line, go to the directory of the source code.

3)Use the following command to create a virtual environment::

  python3 -m venv venv/  
  source venv/bin/activate  
 
We note that creating a virtual environment is not a required step. However, we tested this
procedure to prevent any conflict, and the code runs smoothly.

Installation
~~~~~~~~~~~~

The package can be installed from the .tar file.

1)Go to the directory of the source code (if a user has not done so yet).

2)Use the following command to install the required packages::

 pip install -r requirements.txt

3)From the command line, use the following command to install PUQ::

 python3 setup.py sdist bdist_wheel 
 pip install ./dist/PUQ-0.1.0.tar.gz
 
Once installed, a user should see ``build/`` and ``dist/`` directories created.

Instructions for running the illustrative examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To replicate Figures~1--3, respectively:

1)Go to the ``examples/`` directory.

2)Execute the followings from the command line::

 python3 figure1b.py
 python3 figure2.py
 python3 figure3.py

Running each script should not take more than 60 sec. See the figures (png files) saved under ``examples/`` directory.

Instructions for running the prominent empirical results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instructions are provided to replicate each panel in Figure~6.

To replicate the upper-left panel (banana function), execute the following from the command line::

 python3 figure6.py -funcname banana
 
Running this script takes about 2hrs on a personal Mac laptop. 
Once completed, ``Figure6_banana.png`` is saved under ``examples/`` directory.
 
To replicate the upper-right panel (bimodal function), execute the following from the command line::

 python3 figure6.py -funcname bimodal

Running this script takes about 2hrs on a personal Mac laptop. 
Once completed, ``Figure6_bimodal.png`` is saved under ``examples/`` directory.
 
To replicate the lower-left panel (unimodal function), execute the following from the command line::

 python3 figure6.py -funcname unimodal

Running this script takes about 1hr on a personal Mac laptop. 
Once completed, ``Figure6_unimodal.png`` is saved under ``examples/`` directory.
 
To replicate the lower-right panel (unidentifiable function), execute the following from the command line::

 python3 figure6.py -funcname unidentifiable
 
Running this script takes about 2hrs on a personal Mac laptop. 
Once completed, ``Figure6_unidentifiable.png`` is saved under ``examples/`` directory.
  
Final comments
~~~~~~~~~~~~~~

Type ``deactivate`` from the command line to deactivate the virtual environment if created.

Type ``pip uninstall PUQ`` from the command line to uninstall the package.

