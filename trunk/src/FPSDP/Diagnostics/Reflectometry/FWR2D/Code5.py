"""Treat Code5 format antenna wave pattern data files

This is right now just a very simple version which needs more development in the future.
By Lei Shi, May 6, 2014
"""
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
        
    def read_header(this):
        f = open(this.filename,'r')
        this.params = {}
        for line in f:
            if '!' in line:
                continue
            elif ':' in line:
                words = line.split(':')
                keyword = words[0]
                value = words[1]
                if keyword == 'Datatype':
                    this.params[keyword] = value.strip(' \n')
                elif keyword == 'Wavelength':
                    values = value.split()
                    num = values[0].strip(' ')
                    
                    unit = values[1].strip(' \n')
                    if unit == 'nm':
                        num *= 1e-7
                        this.params[keyword]=num
    
