import collections
import numpy as np

def integration_points(dim, meth, obj='', size=-1):
    """ Return the points and the weight for a 2D integration
        Arguments:
        dim   --  dimension of the integration (int)
        obj   --  type of domain of integration (for dim>1)
        meth  --  method of integration
        size  --  number describing the obj (for example the radius
                  for a circle)
    """
    if dim == 1:
        if meth == 'GL3': # gauss legendre order 3
            w = np.array([1.0,1.0])
            temp = np.sqrt(1.0/3.0)
            points = np.array([-temp,temp])
    elif dim == 2:
        if obj == 'circle':
            if size == -1:
                raise NameError('You should specify a radius')
            if meth == 'order10':
                w = np.zeros(21)
                points = np.zeros((21,2))
                # low and high refere to the radius of the points
                low = np.sqrt(6.0)
                low_weight = (16+low)/360.0
                high_weight = (16-low)/360.0
                high = np.sqrt((6.0+low)/10.0)*size
                low = np.sqrt((6.0-low)/10.0)*size

                w[0] = 1.0/9.0
                for i in range(1,11):
                    w[i] = low_weight
                    angle = 2*np.pi*i/10.0
                    points[i,:] = np.array([low*np.cos(angle),
                                            low*np.sin(angle)])

                    w[i+10] = high_weight
                    points[i+10,:] = np.array([high*np.cos(angle),
                                               high*np.sin(angle)])

    named = collections.namedtuple('Quadrature',['w','pts'])
    return named(w,points) # can be acces with the following way
    # .w or .pts
