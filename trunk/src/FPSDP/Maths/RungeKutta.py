import numpy as np

def runge_kutta_explicit(order,alpha=None):
    """ Coefficient of the explicit Runge-Kutta methods.
    
    :param int order: Order of the Runge-Kutta method
    :returns: Coefficient of the Butcher table
    :rtype: tuple(a,b,c)
    """
    if order == 4:
        a = np.array([[0,0,0,0],[0.5,0,0,0],
                      [0,0.5,0,0],[0,0,1,0]])
        b = np.array([1.0/6.0,1.0/3.0,1.0/3.0,1.0/6.0])
        c = np.array([1,0.5,0.5,0])
    elif order == 3:
        a = np.array([[0,0,0],[0.5,0,0],[-1,2,0]])
        b = np.array([1.0/6.0,2.0/3.0,1.0/6.0])
        c = np.array([1,0.5,0])
    elif order == 2:
        # if alpha == 0.5, mid point rule
        if alpha == None:
            alpha = 0.5
        a = np.array([[0,0],[alpha,0]])
        b = np.array([1-0.5/alpha,0.5/alpha])
        c = np.array([0,alpha])
    elif order == 1:
        # Euler explicit
        a = np.array(0)
        b = np.array(1)
        c = np.array(0)

    return a,b,c
