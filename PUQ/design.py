"""
This module contains a class that implements the main emulation method.
"""
import numpy as np
import importlib
import copy
import warnings


class designer(object):

    def __init__(self,
                 data_cls=None,
                 method='SEQCAL',
                 args={}):
        
        self.method = \
                importlib.import_module('paractive.designmethods.' + method)
        self._info = {'method': method}
        
        self.fit(data_cls, args)


    def fit(self, data_cls, args):

        self.method.fit(self._info, data_cls, args)

