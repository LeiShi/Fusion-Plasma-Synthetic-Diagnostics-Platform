"""Treat Code5 format antenna wave pattern data files

This is right now just a very simple version which needs more development in the future.
By Lei Shi, May 6, 2014
"""
from ....IO.IO_funcs import parse_num

from scipy.interpolate import RectBivariateSpline
import numpy as np

class C5_Error(Exception):
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class C5_reader:
    """Simple reader. 
    """

    def __init__(this,filename):
        """initiates with a file name

        filename: string, the full name of the Code5 file
        """
        this.filename = filename
        this.read_header()
        this.read_Efield()
        this.setup_spline()
        
    def read_header(this):
        f = open(this.filename,'r')
        this.params = {}
        this.comment_line = 0 # count the starting comment lines
        for line in f:
            if '!' in line:
                this.comment_line += 1
                continue
            elif ':' in line:
                words = line.split(':')
                key = words[0]
                value = words[1]
                if key == 'Datatype':
                    this.params['Datatype'] = value.strip(' \n')
                elif key == 'Wavelength':
                    values = value.split()
                    num = parse_num(values[0].strip(' '))
                    
                    unit = values[1].strip(' \n')
                    if unit == 'nm':
                        num *= 1e-7
                        this.params['Wavelength']=num
                    else:
                        raise C5_Error('Default unit for wavelength is nanometer. Please change your code5 file.')
                        
                elif key == 'Grid spacing':
                    values = value.split()
                    dx = parse_num(values[0])
                    dy = parse_num(values[1])
                    unit = values[2].strip(' \n')
                    if unit == 'mm':
                        dx *= 0.1
                        dy *= 0.1
                        spacing = (dx,dy)
                        this.params['Grid_spacing']=spacing
                    else:
                        raise C5_Error('Default unit for grid spacing is milimeter. Please change your code5 file.')
                       
                elif key == 'Coordinates':
                    values = value.split()
                    x0 = parse_num(values[0])
                    y0 = parse_num(values[1])
                    z0 = parse_num(values[2])
                    unit = values[3].strip(' \n')
                    if unit == 'mm':
                        x0 *= 0.1
                        y0 *= 0.1
                        z0 *= 0.1
                        coords = (x0,y0,z0)
                        this.params['Coordinates'] = coords
                    else:
                        raise C5_Error('Default unit for coordinates is milimeter. Please change your code5 file.')
                        
                elif key == 'Direction':
                    values = value.split()
                    x_dir = parse_num(values[0])
                    y_dir = parse_num(values[1])
                    z_dir = parse_num(values[2])
                    dirs = (x_dir,y_dir,z_dir)
                    this.params['Direction'] = dirs
                elif key == 'Array size':
                    values = value.split()
                    nx = parse_num(values[0])
                    ny = parse_num(values[1])
                    this.params['Array_size']=(nx,ny)
                else:
                    raise C5_Error('Unexpected keyword occured:{0}! Please double check the compatibility of the Code5 file version and this program.'.format(key) )
                    
            else:
                print 'header loading finished.'
                break
        f.close()

    def read_Efield(this):
        """read the complex Efield from Code 5 file.
        Note that the result array is in the shape (ny,nx), where ny and nx are contained in this.params['Array_size'], which is read in read_header() method. And each element in the result array is a complex number.
        """
        (nx,ny)= this.params['Array_size']
        data = np.loadtxt(this.filename,comments = '!', skiprows = this.comment_line + 6)
        if(this.params['Datatype'] == 'Complex'):
            data = data.reshape((ny,nx,2))
            this.E_field = data[:,:,0]+ 1j* data[:,:,1]
        else:
            raise C5_Error( 'Right now, only complex data is accepted.')
        
    def setup_spline(this,method = 'RectBivariateSpline'):
        """setup the spline interpolator for outside use

        The default interpolation method is the RectBivariateSpline method.
        Available methods are going to be added in the future if needed.

        result interpolator is named: this.E_re_interp,E_im_interp
        Note that the coordinate
        """

        E_re = np.real(this.E_field)
        E_im = np.imag(this.E_field)

        

        nx,ny = this.params['Array_size']
        x0,y0,z0 = this.params['Coordinates']
        dx,dy = this.params['Grid_spacing']
       
        xmin = x0 - dx*(nx-1.)/2
        xmax = x0 + dx*(nx-1.)/2
        this.X1D = np.linspace(xmin,xmax,nx)
        ymin = y0 - dy*(ny-1.)/2
        ymax = y0 + dy*(ny-1.)/2
        this.Y1D = np.linspace(ymin,ymax,ny)

        if(method == 'RectBivariateSpline'):
            this.E_re_interp = RectBivariateSpline(this.Y1D,this.X1D,E_re)
            this.E_im_interp = RectBivariateSpline(this.Y1D,this.X1D,E_im)

        
