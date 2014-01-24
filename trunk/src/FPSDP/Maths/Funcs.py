"""simple math functions used for debugging and/or productive runs
"""

def my_quad(y,x):
    """quadratic integration on given grids
        I = Sum (y[i+1]+y[i])*(x[i+1]-x[i])/2  
    """
    I = 0.
    for i in range(len(x)-1):
        I += (y[i+1]+y[i])*(x[i+1]-x[i])/2.
        
    return I

def determinent3d(x1,y1,z1,x2,y2,z2,x3,y3,z3):
    """calculate the determinent of 3*3 matrix
    """

    return (x1*y2*z3 + y1*z2*x3 + z1*x2*y3 - x1*z2*y3 - y1*x2*z3 - z1*y2*x3)

