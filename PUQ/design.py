"""
This module contains a class that implements the main design method.
"""

import importlib

class designer(object):
    def __init__(self, data_cls=None, acquisition=None, method="SEQCAL", args={}):
        self.method = importlib.import_module("PUQ.designmethods." + method)
        self._info = {"method": method}

        self.fit(data_cls, acquisition, **args)

    def fit(self, data_cls, acquisition, **args):
        self.method.fit(self._info, data_cls, acquisition, **args)
