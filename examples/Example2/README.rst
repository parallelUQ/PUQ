
Examples
~~~~~~~~

This example demonstrates how to apply the proposed method from Sürer (2024), 
Simulation Experiment Design for Calibration via Active Learning, to a deterministic simulation model with one-dimensional outputs.


Instructions for running the illustrative examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To replicate the figures below, respectively:

1) Go to the ``examples/Example2`` directory.

2) Execute the followings from the command line::

    python example.py

Running this script should not take more than 60 sec. See the figures (png files) saved under the directory.


.. image:: ex2.png
   :alt: Illustration of PUQ with the example
   :align: center
   :width: 600
   

The left panel shows simulation model outputs across design inputs (x) at four 
different parameter values. Black dots represent field data observed at five 
equally spaced design inputs. The right panel displays the points acquired 
using the proposed acquisition function.
