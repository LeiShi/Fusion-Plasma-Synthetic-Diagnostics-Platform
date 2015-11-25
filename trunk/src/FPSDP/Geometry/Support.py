# -*- coding: utf-8 -*-
"""
Contains supportive functions related to geometry and grid

Created on Fri Oct 16 10:35:35 2015

@author: lei
"""
import numpy as np
from matplotlib.tri import TriFinder
from scipy.spatial import Delaunay


# A helper class for using matplotlib.tri.CubicTriInterpolator over a complicated mesh where the default TriFinder doesn't work very well, and scipy.spatial.Delaunay's finder needs to be used.
class DelaunayTriFinder(TriFinder):
    
    def __init__(self,delaunay, triangulation):
        """ Creating a TriFinder for matplotlib.tri.triangulation using the scipy.spatial.Delaunay object
        Compatibility is not checked!
        User must make sure the triangulation is created by the same Delaunay object's *simplices* information, and of course the Delaunay must be of 2-dimensional.
        """
        self.delaunay = delaunay
        super(DelaunayTriFinder,self).__init__(triangulation)
        assert isinstance(delaunay, Delaunay)
        
    def __call__(self,x,y):
        """ find the corresponding simplices (triangles) using Delaunay method: find_simplex(p)
            :param x: x coordinates of specified points
            :type x: numpy array of float
            :param y: y coordinates of specified points
            :type y: numpy array of float
            :return s: indices of triangles within which each point lies.
            :rtype s: numpy array of int
        """
        
        assert x.shape == y.shape
        
        axes = range(1,x.ndim+1)
        axes.append(0)
        
        p = np.array([x,y]).transpose(axes)
        
        return self.delaunay.find_simplex(p)
