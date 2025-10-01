
Examples
~~~~~~~~

This example demonstrates how to apply the proposed method from Sürer, Plumlee, and Wild (2024), 
Sequential Bayesian Experimental Design for Calibration of Expensive Simulation Models, 
to a deterministic simulation model with high-dimensional outputs.


**Instructions for running the illustrative examples**

To replicate the figures below, respectively:

1) Go to the ``examples/Example1`` directory.

2) Execute the followings from the command line::

    python example.py

Running this script should not take more than 120 sec. See the figures (png files) saved under the directory.

.. image:: ex1.png
   :alt: Illustration of PUQ with the example
   :align: center
   :width: 600
   
Blue circles denote the initial design, and plus markers indicate the acquired 
points obtained via random sampling from the prior (left), variance (middle), and proposed 
integrated variance (IVAR) acquisition functions (right).