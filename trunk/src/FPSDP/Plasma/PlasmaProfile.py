# -*- coding: utf-8 -*-
"""
This module defines the plasma profile class.

All profile generators -- analytical, experimental, or simulational-- must 
provide output of plasma profile object, which contains all useful information
of certain synthetic diagnostic requires.

Created on Mon Jan 18 13:27:04 2016

@author: lei
"""

class PlasmaProfile(object):
    """Base class for all plasma profiles required for synthetic diagnostics.
        
    In general, a profile can have no plasma quantities, but must have a grid
    layout to contain possible quantities.
    
    :param grid: Grid for the profiles
    :type grid: :py:class:`..Geometry.Grid.Grid` object
    """
    
    def __init__(self, grid):
        self.grid = grid
        
    def physical_quantities(self):
        return 'none'
        
    def __str__(self):
        return 'General plasma profile:\nGrid:{}\n\
        Physical Quantities:{}'.format(str(self.grid),self.physical_quantities)
        
class ECEI_Profile(PlasmaProfile):
    """Plasma profile for synthetic Electron Cyclotron Emission Imaging.
    
    ECEI needs the following plasma quantities:
    
    :var ne: local electron density
    :var Te_para: local electron temperature parallel to B
    :var Te_perp: local electron temperature perpendicular to B
    :var B: local total magnetic field strength
    
    These should all be passed in compatible with the ``grid`` specification.
    :raises AssertionError: if any of the above quantities are not compatible
    """
    def __init__(self, grid, ne, Te_para, Te_perp, B):
        # test if all quantities has same shape as the grid
        assert ne.shape == grid.shape
        assert Te_para.shape == grid.shape
        assert Te_perp.shape == grid.shape
        assert B.shape == grid.shape
        self.grid = grid
        self.ne = ne
        self.Te_para = Te_para
        self.Te_perp = Te_perp
        self.B = B
        # For some simplified ECEI model, isotropic temperature is assumed.
        self.Te = self.Te_perp

