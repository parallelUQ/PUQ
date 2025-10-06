Release Notes
=============

Below are the release notes for PUQ.

May reference issues on:
https://github.com/parallelUQ/PUQ/issues

Release 0.1.1
-------------

:Date: Oct 8, 2025

- :code:`surrogatemethods` have been updated to use the `hetGPy <https://github.com/davidogara/hetGPy>`_ package as their base.
- :code:`surrogatemethods` provide four emulators for deterministic, stochastic, and one- or multi-dimensional outputs.
- :code:`designmethods` provide four sequential design procedures.
- :code:`examples` have been revised to include illustrative cases from five different papers.
- :code:`tests` include new checks for both emulators and design methods.



Release 0.1.0
-------------

:Date: Sep 10, 2022

Initial release.

Known issues and desired features will be raised on GitHub post-release.

Known issues:

 - create unit tests 
 - create documentation

Desired features:

 - add a funcx simulation interface, using libEnsemble release 0.9
 - new methods stochastic simulation models
 - generalize methods for both design inputs and parameters
