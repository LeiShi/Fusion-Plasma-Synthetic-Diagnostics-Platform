import collections
import numpy as np

def integration_points(dim, meth, obj='', size=-1):
    """ Defines a few quadrature formula (in any number of dimension)

    :param int dim: Dimension of the integration
    :param str obj: Type of domain of integration (for dim>1, e.g. 'disk' )
    :param str meth: Method of integration (e.g 'GL3' for Gauss-Legendre order 3)
    :param size: Object describing the geometry of the problem (e.g. radius for a disk)

    :returns: Points and weights of the quadrature formula
    :rtype: Named tuple (.pts and .w)
    """
    if dim == 1:
        if meth == 'GL3': # gauss legendre order 3
            w = np.array([1.0,1.0])
            temp = np.sqrt(1.0/3.0)
            points = np.array([-temp,temp])

        elif meth == 'GL5':
            points = np.zeros(3)
            w = np.zeros(3)
            w[0] = 8.0/9.0
            w[1] = 5.0/9.0
            w[2] = w[1]

            points[0] = 0
            points[1] = np.sqrt(3.0/5.0)
            points[2] = -points[1]
            
        elif meth == 'GL7':
            temp = (2.0/7.0)*np.sqrt(6.0/5.0)
            points = np.zeros(4)
            w = np.zeros(4)
            points[0] = np.sqrt(3.0/7.0-temp)
            points[1] = -points[0]
            points[2] = np.sqrt(3.0/7.0+temp)
            points[3] = -points[2]

            temp = np.sqrt(30.0)
            w[0] = (18.0+temp)/36.0
            w[1] = w[0]
            w[2] = (18.0-temp)/36.0
            w[3] = w[2]

        elif meth == 'GL19':
            w = np.zeros(10)
            points = np.zeros(10)
            # from http://pomax.github.io/bezierinfo/legendre-gauss.html
            w[0] = 0.2955242247147529
            w[1] = 0.2955242247147529
            w[2] = 0.2692667193099963
            w[3] = 0.2692667193099963
            w[4] = 0.2190863625159820
            w[5] = 0.2190863625159820
            w[6] = 0.1494513491505806
            w[7] = 0.1494513491505806
            w[8] = 0.0666713443086881
            w[9] = 0.0666713443086881

            points[0] = -0.1488743389816312
            points[1] = 0.1488743389816312
            points[2] = -0.4333953941292472
            points[3] = 0.4333953941292472
            points[4] = -0.6794095682990244
            points[5] = 0.6794095682990244
            points[6] = -0.8650633666889845
            points[7] = 0.8650633666889845
            points[8] = -0.9739065285171717
            points[9] = 0.9739065285171717
    elif dim == 2:
        if obj == 'disk':
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
