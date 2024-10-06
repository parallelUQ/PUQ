
Examples
~~~~~~~~

These examples replicate the results presented in the paper titled 'Sequential Bayesian 
Experimental Design for Calibration of Expensive Simulation Models' by Sürer, Plumlee, and Wild (2024).

**Instructions for running the illustrative examples**

To replicate Figures~1--3, respectively:

1) Go to the ``examples/Technometrics2024`` directory.

2) Execute the followings from the command line::

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
