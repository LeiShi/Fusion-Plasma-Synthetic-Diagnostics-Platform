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