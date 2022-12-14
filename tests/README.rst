Running Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``PUQ`` uses GitHub Actions that allow to set up Continuous Integration workflows.
After each push and pull request to develop and master branch, the code will be tested
with automated workflows, and the Python unit tests under ``tests/unit_tests``
will automatically be running. Reviewer will check that all tests have passed and will then approve merge.

The test suite requires the pytest_ and pytest-cov_ packages to be installed and
can all be run from the ``tests/`` directory of the source distribution by
running::

./run-tests.sh

Coverage reports are produced under the relevant directory only if all tests are
used.

If you have the source distribution, you can run the tests in the top-level
directory containing the setup script with ::

 python setup.py test
