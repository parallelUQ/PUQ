Examples
~~~~~~~~

These examples replicate the results presented in the paper titled 'Performance Analysis of 
Sequential Experimental Design for Calibration in Parallel Computing Environments' 
by Sürer and Wild (2024).

**Instructions for running the illustrative examples with performance model**

To replicate Figures~3 and 7--10:

1) Go to the ``examples/`` directory.

2) Execute any of the following from the command line:

.. code-block:: python

 python3 Fig3.py
 python3 Fig7.py
 python3 Fig8.py
 python3 Fig9.py
 python3 Fig10.py
 
Running each script should not take more than 90 sec. See the figures (jpeg files) saved under ``examples/`` directory.

**Instructions for running the sequential design using different synthetic simulation models**

To replicate Figure~2, execute the following from the command line:

.. code-block:: python

 python3 Fig2.py
 
Running this script takes about a minute on a personal Mac laptop. 

To collect ``-max_eval`` simulation outputs from parameters acquired with 
different acquisition functions (``-al_func``) and synthetic simulation models (``-funcname``), 
one can use the script ``run_test_funcs.py``.

As an example, execute any of the following from the command line:

.. code-block:: python

 python3 run_test_funcs.py -funcname ackley -al_func ei -max_eval 200
 python3 run_test_funcs.py -funcname easom -al_func ei -max_eval 200
 python3 run_test_funcs.py -funcname himmelblau -al_func eivar -max_eval 200
 python3 run_test_funcs.py -funcname holder -al_func eivar -max_eval 200
 python3 run_test_funcs.py -funcname matyas -al_func hybrid_ei -max_eval 200
 python3 run_test_funcs.py -funcname sphere -al_func hybrid_ei -max_eval 200
 
Once completed, ``Figure_funcname.jpg`` is saved under ``examples/`` directory.
Running each script should not take more than one minute on a personal Mac laptop.