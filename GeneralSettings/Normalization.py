# normalization plans and functions

# Module depends on PhysConst ( Temporarily not necessary ) 
# from .PhysConst import CurUnitSys as cu

# default, no normalization in cgs
Norm={'L':1, 'M':1, 'T':1}

# set up customized normalization
def set_normalization(**N):
    if ( 'L' in N.keys() and 'm' in N.keys() and 't' in N.keys() ):
        for k in N.keys():
            Norm[k] = N[k]
    else:
        # exception handling may be needed
        print ' Normalization list not valid. At least contain "L", "m", and "t" keys. '

