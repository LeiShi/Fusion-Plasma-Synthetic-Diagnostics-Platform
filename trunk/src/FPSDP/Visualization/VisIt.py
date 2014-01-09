import numpy as np

class VisItError(Exception):
    def __init__(this,*p):
        this.args = p
        
class Point:
    """class of single grid point
    Attributes:
        X,Y,Z: double, coordinates of the point 
        DataNames: list of string, the name list of all quantities associate with this point
        Data: list of double, the value corresponding the names
    """
    def __init__(this, x, y, z, **data):
        """create a Point object based on the x,y,z components and other attributes given by keyword arguments
        x,y,z: double, cartesian coordinates of the point
        keywords: "key = value" format, key will be stored as the name of the value 
        """
        this.X = x
        this.Y = y
        this.Z = z
        this.DataNames = [s for s in data.keys()]
        this.Data = [v for v in data.values()]
        

class Polygon:
    """class of single polygon
    Attributes:
        VertLen: int, number of vertices, at least 3, normally 4
        Vertices: list of Point, 3 or more points that construct the polygon
        VerticeNumbers: list of int, the number of each node in the mesh node list
        DataNames: list of string, the names of data stored on this polygon cell
        Data: list of double, the value corresponding the names 
    """
    def __init__(this, points, pointnumbers, **data):
        """create a polygon cell object
        
        points: list of Point, at least 3 normally 4 points should be given, which are the vertices of polygon
        pointnumbers: list of int, the number of the points in the whole mesh point list, as the same order as those given by the first argument
        """
        if (len(points)!= len(pointnumbers)):
            raise(VisItError('Creating Polygon Error: length of points and pointnumbers mismatch!'))
        this.VertLen = len(points)
        this.Vertices =[points[i] for i in range(this.VertLen)]
        this.VerticeNumbers = [pointnumbers[i] for i in range(this.VertLen)]
        this.DataNames = [s for s in data.keys()]
        this.Data = [v for v in data.values()]
        
class Mesh:
    """class of a whole mesh
    Attributes:
        PointList: list of Point, All the grid points in the mesh
        PolygonList: list of Polygon, All the polygons in the mesh
    Methods:
        create_vtk_file: print out .vtk files
        make_surface: make 3D surface

        make_uniform_surface: make 3D uniform surface mesh out of range and resolution for x,y direction, with a specified z on each (x,y)  
    """
    def __init__(this, points, polygons):
        this.PointList = [points[i] for i in range(len(points))]
        this.PolygonList = [polygons[i] for i in range(len(polygons))]
        this.PolyDataLen = sum([this.PolygonList[i].VertLen+1 for i in range(len(this.PolygonList))])
        
    def output_vtk(this, fname = 'TemporaryMesh.vtk', ftag = 'temp file', code = 'ASCII', dataset = 'POLYDATA', datatype = 'ALL'):
        """Write the mesh into a .vtk file for VisIt use
        fname: string, specify the output file name if you want to keep the record
        ftag: string, file name string inside the file
        code: string, code type of the file, default is ASCII
        dataset: string, data format type, default is POLYDATA
        datatype: string, data to be output, default is ALL, can be set as 'POINT_ONLY' or 'CELL_ONLY'
        """
        
        with open(fname,'w') as f:
            f.write('# vtk DataFile Version 3.0\n')
            f.write(ftag+'\n')
            f.write(code+'\n')
            f.write('DATASET '+dataset+ '\n')
            f.write('POINTS ' + str(len(this.PointList)) +' double\n')
            for p in this.PointList:
                f.write(str(p.X) + ' ' + str(p.Y) + ' '+ str(p.Z) + '\n')
            
            f.write('POLYGONS ' + str(len(this.PolygonList)) + ' ' + str(this.PolyDataLen) + '\n')
            for p in this.PolygonList:
                f.write(str(p.VertLen)+ ' ')
                for v in p.VerticeNumbers:
                    f.write(str(v) + ' ')
                f.write('\n')
            if(datatype == 'ALL'):
                if( this.PointList[0].DataNames ):
                    f.write('POINT_DATA ' + str(len(this.PointList)) + '\n') 
                    for i in range(len(this.PointList[0].DataNames)):       
                        f.write('SCALARS ' + this.PointList[0].DataNames[i] +' double 1\n')
                        f.write('LOOKUP_TABLE default\n')
                        for p in this.PointList:
                            f.write(str(p.Data[i]) +'\n')
                if( this.PolygonList[0].DataNames ): 
                    f.write('CELL_DATA ' + str(len(this.PolygonList)) + '\n')
                    for i in range(len(this.PolygonList[0].DataNames)):       
                        f.write('SCALARS ' + this.PolygonList[0].DataNames[i] +' double 1\n')
                        f.write('LOOKUP_TABLE default\n')
                        for p in this.PolygonList:
                            f.write(str(p.Data[i]) +'\n')
            else:
                pass
                
#### Finish Defining Objects #############################################################################
#### Start Writing top level handlers ####################################################################

def make_square_surface(X1D, Y1D, z = np.zeros((1,1)), **data):
    """Make 3D surface mesh out of 1D values of X,Y and a specified z values on each (x,y)
    Arguments:
        X1D,Y1D: 1D array of double, the 1D values in x,y coordinates
        **data: expecting keyword list of all data stored on grid points, the keyword will be the name of the data entry, and value should be a 3D array with all the data stored in format [Y,X]. Note that a 'z' keyword is required to create the 3D surface mesh.
    return:
        Mesh object
    """
    NX = len(X1D)
    NY = len(Y1D)

    if ( z.shape != (NY,NX) ):
        print 'WARNING: Surface is made without specifying z values, the result will be a 2D flat mesh.'
        z = np.zeros( (NY,NX) )        
    DataNames = data.keys()
    DataLen = len(DataNames)
    Data = data.values()
    
    PointDataNames = []
    PointData = []
    CellDataNames = []
    CellData = []
    
    for i in range(DataLen):
        if(Data[i].shape == (NY,NX)):
            PointDataNames.append(DataNames[i])
            PointData.append(Data[i])
        elif(Data[i].shape == (NY-1,NX-1)):
            CellDataNames.append(DataNames[i])
            CellData.append(Data[i])
        else:
            raise VisItError('make square mesh error: some data given is neither point-like nor cell-like. Note that the data structure is ordered in (Y,X) form!')
    
    
    points = []
    polygons = []
    for i in range(NY):
        for j in range(NX):                
            newdata = dict((PointDataNames[p],PointData[p][i,j]) for p in range(len(PointDataNames)) ) 
            points.append(Point(X1D[j],Y1D[i],z[i,j], **newdata))
    for i in range(NY-1):
        for j in range(NX-1):
            lowest = i*NX+j #the lowest indice of the needed point
            polypnumbers = [lowest, lowest+1, lowest+NX+1, lowest+NX]
            polypoints = [points[lowest],points[lowest+1],points[lowest+NX+1], points[lowest+NX]]
            newdata = dict( (CellDataNames[p],CellData[p][i,j]) for p in range(len(CellDataNames)) )
            polygons.append(Polygon(polypoints,polypnumbers,**newdata))
    return Mesh(points,polygons)

import scipy.io.netcdf as nc             
                                    
def load_profile_from_netcdf_fluctuation(fname):
    """read plasma profile from fluctuation netcdf files, return the mesh object created based on r,z coordinates defined in the file
    """
    f = nc.netcdf_file(fname,'r')
    x = f.variables['rr'].data
    y = f.variables['zz'].data
    Y,X = np.meshgrid(x,y)
    ne = f.variables['ne'].data
    
    z = ne / np.max(ne) 
    
    te = f.variables['te'].data
    B = f.variables['bb'].data
    
    mesh = make_square_surface(x,y,z,ne = ne, te = te, B = B)
    f.close()
    return mesh
    
    
def load_paraxial_from_netcdf(fname):
    f = nc.netcdf_file(fname,'r')
    
    nx = f.dimensions['p_nx']
    ny = f.dimensions['p_ny']
    x = f.variables['p_x'].data
    y = f.variables['p_y'].data
    Er = f.variables['p_Er'].data
    Ei = f.variables['p_Ei'].data
    
    f.close()
    
    #rescale to meter
    x /= 100
    y /= 100 
    
    z = np.ones((ny,nx))
    
    Er_in = Er[0,:,:]
    Er_ref = Er[1,:,:]
    Ei_in = Ei[0,:,:]
    Ei_ref = Ei[1,:,:]
    
    Er = Er_in + Er_ref
    Ei= Ei_in + Ei_ref
    Esq = Er**2 + Ei**2
    
    return make_square_surface(x,y,z=z,Er = Er, Ei=Ei,Esq = Esq)
    
def load_fullwave_from_netcdf(fname):
    f = nc.netcdf_file(fname,'r')
    
    nx = f.dimensions['s_nx']
    ny = f.dimensions['s_ny']
    x = f.variables['s_x'].data
    y = f.variables['s_y'].data
    Er = f.variables['s_Er'].data
    Ei = f.variables['s_Ei'].data
    
    f.close()
    
    x /= 100
    y /= 100
    
    z = np.ones((ny,nx))
    
    Esq = Er**2 + Ei**2
    
    return make_square_surface(x,y,z=z,Er = Er, Ei=Ei,Esq = Esq)
    
    
    
    
    
    
    
       
   
                   
                     
        