# -*- coding: utf-8 -*-
"""
Abstract class :py:class:`Fluctuation`. :py:class:`Eigenmode` and :py:class:`Turbulence` are derived from it. 

Created on Thu Oct 08 17:41:51 2015

@author: lei
"""

from abc import ABCMeta, abstractmethod, abstractproperty
from ...Geometry.Grid import Grid
class Fluctuation(object):
    """Abstract base class for plasma fluctuations
    Two classes are derived from this class: :py:class:`Eigenmodes`, :py:class:`Turbulence`
    
    They both substantiate the following methods:
        realize([t=None, grid=None]): generate a realization for given time(*t*, optional) and space grid (*grid*, optional). If t or grid not given, then the internal *time* and *grid* will be used. A float array as same shape as grid should be returned containing the fluctuation data.
    
    Notes:
    A fluctuation is a time and space dependent variation of some physical quantities on top of their mean value -- the equilibrium. 
    In plasmas, fluctuations can mainly be devided into two catagories: superposition of eigenmodes, and turbulences. 
    Normally, the eigenmodes state is so-called linear or quasi-linear, because the direct mode-mode coupling is negligible. 
    On the other hand, turbulence state is strongly non-linear, so no well defined mode structure can be found there. 
    Therefore, the ways to describe these two kind of fluctuations are quite different. 
        For eigenmodes, we can specify the mode structure for each dominant mode. 
            For example, to specify a global mode, we can write down the envolope of the amplitude as a function of *r*, and the mode numbers in poloidal and toloidal direction, *m* and *n* respectively.
            If we are looking for the time dependent behavior, we should also specify the mode frequency, :math:`\omega`. Sometimes this information is calculated by other codes, sometimes we will just use some analytic model.
        For turbulence, what we know is usually just some statistical characteristics of the fluctuation, and we rely on randomly generated *realizations* of the fluctuations to numerically study the statistics of some resulting phenomena, e.g. diagnostic results.
            In this case, we normally assume some kind of *auto-correlation function* or *power spectrum* of the turbulence, and create random realizations based on that. A more detailed discussion can be found in :py:module:`Turbulence`.
    
    """
    __metaclass__ = ABCMeta
    
    def __init__(self, grid=None):

        self._grid = grid
        
    @abstractmethod
    def realize(self, t=None, grid = None):
        if(t == None):
            t = self.t + self.dt
            self.t = t
        if(grid == None):
            grid = self.grid
        #Create the realization Here   
        return self.realize(t,grid)
        
        
    @property
    def grid(self):
        """Default spatial grid
        """
        return self._grid
        
    @grid.setter
    def grid(self,value):
        if isinstance(value,Grid):
            self._grid = value
        else:
            raise ValueError('Wrong type of grid. Need to be a subclass of Geometry.Grid')
            
    @grid.deler
    def grid(self):
        del self._grid
        
    
        
    
    