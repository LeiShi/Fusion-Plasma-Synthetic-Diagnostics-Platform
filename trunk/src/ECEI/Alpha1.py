"""Module contains Functions that calculate the local absorption coefficient alpha.
"""

#module depends on scipy package which automatically imports numpy
import scipy as sp
#rename numpy as np for convention
import numpy as np

def create_f72( Grid ):
    """create the F7/2 table on grid points
    """

