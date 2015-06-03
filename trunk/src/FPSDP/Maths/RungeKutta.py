import numpy as np

def runge_kutta_explicit(order,alpha=1):
    """ Coefficient of the explicit Runge-Kutta methods.
    
    :param int order: Order of the Runge-Kutta method
    :returns: Coefficient of the Butcher table and the number of stage
    :rtype: tuple(a,b,c,Nstage)
    """
    if order == 4:
        a = np.array([[0,0,0,0],[0.5,0,0,0],
                      [0,0.5,0,0],[0,0,1,0]])
        b = np.array([1.0/6.0,1.0/3.0,1.0/3.0,1.0/6.0])
        c = np.array([1,0.5,0.5,0])
        Nstage = 4
    elif order == 3:
        a = np.array([[0,0,0],[0.5,0,0],[-1,2,0]])
        b = np.array([1.0/6.0,2.0/3.0,1.0/6.0])
        c = np.array([1,0.5,0])
        Nstage = 3
    elif order == 2:
        # if alpha == 0.5, heun's method
        # if alpha == 1, midpoint method
        a = np.array([[0,0],[alpha,0]])
        b = np.array([1-0.5/alpha,0.5/alpha])
        c = np.array([0,alpha])
        Nstage = 2
    elif order == 1:
        # Euler explicit
        a = np.array([[0]])
        b = np.array([1])
        c = np.array([0])
        Nstage = 1

    return a,b,c,Nstage
