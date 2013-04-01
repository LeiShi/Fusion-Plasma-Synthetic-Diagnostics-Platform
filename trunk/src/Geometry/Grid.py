"""Class definitions of grids 
"""
#moudle depends on numpy package
import numpy as np

class GridError(Exception):
    def __init__(this,*p):
        this.args = p

class Grid:
    """Base class for all kinds of grids

    contains name of the grid
    """
    def __init__(this):
        this._name = 'General Grids'

    def tell(this):
        return this._name

class Cartesian2D(Grid):
    """Cartesian grids in 2D space. Generally corresponds a toroidal slice, i.e. R-Z plane. Rectangular shape assumed.

    Attributes:
        float ResR: the resolution in R direction. 
        float ResZ: the resolution in Z direction.
        point DownLeft: the coordinates of the down left cornor point, given in (Zmin,Rmin) value pair form.
        point UpRight: the coordinates of the up right cornor point, given in (Zmax,Rmax) form.
        int NR,NZ: The gird number in R,Z directions. Can be specified initially or derived from other parameters.
        2darray R2D: R values on 2D grids. R(0,:) gives the 1D R values.
        2darray Z2D: Z values on 2D grids. R is the fast changing variable so Z2D(:,0) gives the 1D Z values.

        
    """
    def __init__(this, **P):
        """initialize the cartesian grid object.

        If either DownLeft or UpRight is not specified, a GridError exception will be raised.
        Either NR or ResN can be specified. If none or both, a GridError exception will be raised. Same as NZ and ResZ
        """
        this._name = '2D Cartesian Grids'
        try:
            if ( 'DownLeft' in P.keys() and 'UpRight' in P.keys() ):
                this.DownLeft ,this.UpRight = P['DownLeft'], P['UpRight']
                Zmax,Rmax = this.UpRight
                Zmin,Rmin = this.DownLeft
                rangeR = float(Rmax - Rmin)
                rangeZ = float(Zmax - Zmin)
                if ( 'NR' in P.keys() and not 'ResR' in P.keys() ):
                    this.NR = P['NR']
                    this.ResR = rangeR / this.NR                
                elif ('ResR' in P.keys() and not 'NR' in P.keys() ):
                    this.NR = int ( rangeR/P['ResR'] + 2 ) # make sure the actual resolution is finer than the required one
                    this.ResR = rangeR / this.NR
                else:
                    raise GridError('NR and ResR missing or conflicting, make sure you specify exactly one of them.')
                if ( 'NZ' in P.keys() and not 'ResZ' in P.keys() ):
                    this.NZ = P['NZ']
                    this.ResZ = rangeZ / this.NZ                
                elif ('ResZ' in P.keys() and not 'NZ' in P.keys() ):
                    this.NZ = int ( rangeZ/P['ResZ'] + 2 ) # make sure the actual resolution is finer than the required one
                    this.ResZ = rangeZ / this.NZ
                else:
                    raise GridError('NZ and ResZ missing or conflicting, make sure you specify exactly one of them.')
            else:
                raise GridError("Initializing Grid fails: UpLeft or DownRight not set.")
        except GridError:
            raise
        except:
            print 'Unexpected error in grid initialization! During reading and comprehensing the arguments.'
            raise

        #now create the 2darrays for R and Z
        this.R2D = np.zeros( (this.NZ,this.NR) ) + np.linspace(Rmin,Rmax,this.NR)[ np.newaxis , : ]
        this.Z2D = np.zeros( (this.NZ,this.NR) ) + np.linspace(Zmin,Zmax,this.NZ)[ : , np.newaxis ]    

    def tell(this):
        """returns the key informations of the grids
        """
        info = this._name + '\n'
        info += 'DownLeft :' + str(this.DownLeft) +'\n'
        info += 'UpRight :' + str(this.UpRight) +'\n'
        info += 'NR,ResR :' + str( (this.NR,this.ResR) ) +'\n'
        info += 'NZ,ResZ :' + str( (this.NZ,this.ResZ) ) +'\n'
        return info