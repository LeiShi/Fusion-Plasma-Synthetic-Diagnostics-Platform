# normalization plans and functions

# Module depends on PhysConst 
from .PhysConst import CurUnitSys as cu

# default values in cgs
Norm={'L':1, 'M':1, 'T':1}

# pre-defined plans
NormPlan = {'ElecScheme':{'L':cu},}
