""" Normalization Plans and Functions
"""

# Module depends on PhysConst ( Temporarily not necessary ) 
# from .PhysConst import CurUnitSys as cu

class NormalizationError(Exception):
    """Exception class in Normaliztion.

    Attributes:
        str: a string contains error information
    """
    def __init__(this, *a):
        """Initiate the exception with a string given or using a default string
        """
        this.args = a

class Norm:
    """Normalization class. Each object should stand for a chosen normalization convention.

    Attributes:
        _data: dictionary contains the values of each normalization quantity
        set_normalization: methed sets the values after Norm opject created
        tell: method returns _data
    """

    # initialized with default, no normalization in cgs, or using a key word argument list
    def __init__(this, **N):
        """Initializing with default normalization, L, m, t set to 1. OR using keyword arguments

        valid keywords:
            L: length normalization
            m: mass normaliztion
            t: time normalization
            [ T: temperature normalization. Since temperature usually uses keV as unit, it's normalization is not very common.]
            [ l: luminous intensity. Not frequently used.]
        """
        this._data = dict(L=1, m=1, t=1)
        for key in N.keys():
            this._data[key]= N[key]
            
        

    
    # set up customized normalization
    def set_normalization(this,**N):
        """ Let users modify the normalization after creating a Norm object. It's usually dangerous. Use this method carefully.
        """
        if ( 'L' in N.keys() and 'm' in N.keys() and 't' in N.keys() ):
            for k in N.keys():
                this._data[k] = N[k]
        else:
            # exception handling may be needed
            raise NormalizationError(' Normalization list not valid. At least contain "L", "m", and "t" keys. ')

    def tell(this):
        """ Show the content of this object, return the _data dictionary.
        """
        return this._data


