# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 10:16:21 2015

@author: lei
"""

from .Fluctuation import Fluctuation

class Eigenmode(Fluctuation):
    
    def __init__(self, t=0, dt=1, grid=None):
        pass
    
    @property
    def t(self):
        """Current internal time
        """
        return self._t
        
    @t.setter
    def t(self,value):
        self._t = value
        
    @t.deler
    def t(self):
        del self._t
        
    @property
    def dt(self):
        """Default time incrementation between each call of realization
        """
        return self._dt
        
    @dt.setter
    def dt(self,value):
        self._dt = value
        
    @dt.deler
    def dt(self):
        del self._dt