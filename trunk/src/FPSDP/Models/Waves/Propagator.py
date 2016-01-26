# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 15:20:15 2016

@author: lei

Propagators for electromagnetic waves propagating in plasma
"""

from abc import ABCMeta, abstractmethod, abstractproperty

class Propagator(object):
    
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def propagate(self, E_start, x_start, x_end):
    
    @abstractproperty
    def dielectric()    
    
    

class ParaxialSolver(object):

