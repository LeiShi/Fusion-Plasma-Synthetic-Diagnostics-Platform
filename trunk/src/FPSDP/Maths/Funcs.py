"""simple math functions used for debugging and/or productive runs
"""
import scipy as sp
import numpy as np
import FPSDP.Maths.Integration as integ


def heuman(phi,m):
    r""" Compute the Heuman's lambda function

    :math:`\Lambda_0 (\xi,k) = \frac{2}{\pi}\left[E(k)F(\xi,k') + K(k)E(\xi,k')- K(k)F(\xi,k')\right]`
    where :math:`k' = \sqrt{(1-k^2)}`

    :param np.array[N] phi: The amplitude of the elliptic integrals
    :param np.array[N] m: The parameter of the elliptic integrals

    :returns: Evaluation of the Heuman's lambda function
    :rtype: np.array[N]
    """
    m2 = 1-m
    F2 = sp.special.ellipkinc(phi,m2) # incomplete elliptic integral of 1st kind
    K = sp.special.ellipk(m) # complete elliptic integral of 1st kind
    E = sp.special.ellipe(m) # complete elliptic integral of 2nd kind
    E2 = sp.special.ellipeinc(phi,m2) # incomplete elliptic integral of 2nd kind
    ret = 2.0*(E*F2+K*E2-K*F2)/np.pi
    return ret

def solid_angle_disk(pos,r):
    r""" Compute the solid angle of a disk on/off-axis from the pos
    the center of the circle should be in (0,0,0)

    .. math::
      \Omega = \left\{\begin{array}{lr}
      2\pi-\frac{2L}{R_\text{max}}K(k)-\pi\Lambda_0(\xi,k) & r_0 < r_m \\
      \phantom{2}\pi-\frac{2L}{R_\text{max}}K(k) & r_0 = r_m \\
      \phantom{2\pi}-\frac{2L}{R_\text{max}}K(k)+\pi\Lambda_0(\xi,k) & r_0 > r_m \\
      \end{array}\right.

    Read the paper of `Paxton  <http://scitation.aip.org/content/aip/journal/rsi/30/4/10.1063/1.1716590>`_ "Solid Angle Calculation for a 
    Circular Disk" in 1959 for the exact computation.

    :param np.array[N,3] pos: Position from which computing the solid angle
    :param float r: Radius of the disk (the disk is centered in (0,0,0) and the perpendicular is along the z-axis)

    :returns: Solid angle for each positions
    :rtype: np.array[N]
    """
    pos = np.atleast_2d(pos)
    # define a few value (look Paxton paper for name)
    r0 = np.sqrt(np.sum(pos[:,0:2]**2, axis=1))
    ind1 = r0 != 0
    ind2 = ~ind1
    Rmax = np.sqrt(pos[ind1,2]**2 + (r0[ind1]+r)**2)
    R1 = np.sqrt(pos[ind1,2]**2 + (r0[ind1]-r)**2)
    k = np.sqrt(1-(R1/Rmax)**2)
    LK_R = 2.0*abs(pos[ind1,2])*sp.special.ellipk(k**2)/Rmax
    # not use for r0=r but it should not append
    # often
    xsi = np.arctan(abs(pos[ind1,2]/(r-r0[ind1])))
    pilam = np.pi*heuman(xsi,k**2)
    # the three different case
    inda = r0[ind1] == r
    indb = r0[ind1] < r
    indc = (~inda) & (~indb)
    # compute the solid angle
    solid = np.zeros(pos.shape[0])
    temp = np.zeros(np.sum(ind1))
    temp[inda] = np.pi - LK_R[inda]
    temp[indb] = 2.0*np.pi - LK_R[indb] - pilam[indb]
    temp[indc] = - LK_R[indc] + pilam[indc]
    solid[ind1] = temp

    # on axis case (easy analytical computation)
    solid[ind2] = 2*np.pi*(1.0 - np.abs(pos[ind2,2])/np.sqrt(r**2 + pos[ind2,2]**2))
    if (solid <= 0).any():
        print('Solid angle:',solid)
        print('Position:', pos)
        raise NameError('Solid angle smaller than 0')
    return solid


def compute_threshold_solid_angle(x,y,pos,rx,ry):
    """ Compute a normalization of the threshold for the function :func:`solid_angle_seg <FPSDP.Maths.Funcs.solid_angle_seg>`

    :param list[x1,x2] x: Intersection on the ring of the mixed case
    :param list[y1,y2] y: Intersection on the lens of the mixed case
    :param np.array[N] pos: Position from where the solid angle is computed
    :param int rx: Radius of the ring
    :param int ry: Radius of the lens

    :return: Threshold
    :rtype: np.array[N]
    """

    x1 = x[0]
    x2 = x[1]

    # compute the scalar product between x1 and x2
    angle = np.einsum('ij,ij->i',x1,x2)
    # index where we compute the area of the smallest part of the disk
    ind = np.einsum('ij,ij->i',pos[...,:2],x1) < 0
    # area of the triangle between O,x1,x2
    Atri = 0.5*angle
    # Compute the angle between x1,0,x2
    angle = np.arccos(angle/rx**2)
    # rectification of the angle for being in [0,2pi]
    angle[ind] = 2*np.pi - angle[ind]
    # approximation of the solid angle for full circle - segment
    A = ((np.pi-0.5*angle)*rx**2 + Atri)/pos[...,2]**2
    
    # same but with y
    y1 = y[0]
    y2 = y[1]

    # compute the scalar product between y1 and y2
    angle = np.einsum('ij,ij->i',y1,y2)
    # index where we compute the area of the smallest part of the disk
    ind = np.einsum('ij,ij->i',pos[...,:2],y1) < 0
    # area of the triangle between O,y1,y2
    Atri = 0.5*angle
    # Compute the angle between y1,0,y2
    angle = np.arccos(angle/ry**2)
    # rectification of the angle for being in [0,2pi]
    angle[ind] = 2*np.pi - angle[ind]
    # approximation of the solid angle for full circle - segment
    A = ((np.pi-0.5*angle)*ry**2 + Atri)/pos[...,2]**2

    return A

    

def solid_angle_seg(pos,x,r,islens,Nth,Nr):
    r""" Compute the solid angle of a disk where a segment has been removed.
    
    First, the numerical integration will be carried out over the biggest area of the disk,
    and, in a second time, if necessary, the integral over the full disk is computed
    (with the analytical formula) and subtracted by the numerical integral.
    When we want to compute the small area with this methods, the error can be bigger than
    the solid angle, therefore an external check need to be done (usually this method is used in 
    a computation in two step with the other one that will be a lot bigger than this error)
    
    The idea is to compute numerically the 2D integral by splitting the domain in 
    sector of the same angle and doing a Gauss-Legendre quadrature formula over
    each dimension.
    
    In a first time, the maximum radius (that will depends on the coordinate :math:`\theta`)
    has to be compute.
    
    WARNING: This function assumed that all the points are at the same distance of the focus points.

    In this figure, we want to compute the area between the black line and the blue one.
    
    .. tikz::
       \draw [red,dashed,domain=115:180] plot ({6*cos(\x)}, {6*sin(\x)});
       \draw [red,dashed,domain=360:425] plot ({6*cos(\x)}, {6*sin(\x)});
       \draw [black,thick,domain=150:390] plot ({3*cos(\x)}, {4+3*sin(\x)});
       \draw [red,thick,domain=65:115] plot ({6*cos(\x)}, {6*sin(\x)});
       \draw [black,dashed,domain=30:150] plot ({3*cos(\x)}, {4+3*sin(\x)});
       \draw [domain=-10:80] plot ({0.8*cos(\x)}, {4+0.8*sin(\x)});
       \node at (1,4.6) {$\theta$};
       \node at (-5,0) {Lens};
       \node at (2.4,1) {Ring};
       \node at (0,0) {x};
       \node at (0,4) {x};
       \draw (0,4) -- (0.51,6.94);
       \draw (0,4) -- ({3*cos(-10)}, {4+3*sin(-10)});
       \node at (2.66,5.38) {x};
       \node at (3.2,5.8) {$x_2$,$y_2$};
       \node at (-2.66,5.38) {x};
       \node at (-3.2,5.8) {$x_1$,$y_1$};
       \draw [blue] (-2.66,5.38) -- (2.66,5.38);
       \node at (0.25,5.4) {x};
       \node at (0.8,5.2) {$r_{max}$};

        
    :todo: improvement: remove useless computation of rmax
    :param np.array[N,3] pos: Position in the optical system
    :param list[np.array[N],..] x: Position of the intersection on the ring (list contains 2 elements) 
    :param float r: Radius of the disk (should be centered at (0,0,0) and the perpendicular should be along the z-axis)
    :param float islens: None if it is not the lens, otherwise the distance between the lens and the focus point
    :param int Nth: Number of sections for the theta quadrature formula
    :param int Nr: Number of sections for the radial quadrature formula

    :return: Solid angle
    :rtype: np.array[N]
    """
    
    # split the two intersections in two variables
    x1 = x[0]
    x2 = x[1]
    
    # limits (in angle) considered for the integration
    theta = np.linspace(0,2*np.pi,Nth)
    quadr = integ.integration_points(1,'GL4') # Gauss-Legendre order 4
    quadt = integ.integration_points(1,'GL4') # Gauss-Legendre order 4
    
    # mid point of the limits in theta
    av = 0.5*(theta[:-1] + theta[1:])
    # half size of the intervals in theta
    diff = 0.5*np.diff(theta)
    # array containing all the value of theta that will be computed
    th = diff[:,np.newaxis]*quadt.pts + av[:,np.newaxis]
    
    # perpendicular vector to x1->x2
    perp = np.copy(-pos[:,:2])
    
    # indices where we want to compute the big part
    ind = np.einsum('ij,ij->i',perp,x1) > 0

    # create a vector perpendicular to x1-x2 and goes to the line from the origin
    perp[~ind] = -perp[~ind]
    perp = perp/np.sqrt(np.sum(perp**2,axis=1))[:,np.newaxis]
    # in the case of the lens and between the fiber and the lens, we want the opposite case
    if islens != None:
        test = pos[:,2] < islens
        ind[test] = ~ind[test]
    
    # unit vector for each angle
    delta = np.array([np.cos(th),np.sin(th)])
    delta = np.rollaxis(delta,0,3)
    # now detla[Nth-1,quadt,dim]
    
    # compute the scalar product (=> the cos)
    cospsi = np.einsum('ak,ijk->aij',perp,delta)
    # index where the line can cross the segment
    ind2 = cospsi > 0
    
    # distance between line
    d = np.abs(x1[:,0]*x2[:,1]-x2[:,0]*x1[:,1])/np.sqrt(np.sum((x2-x1)**2,axis=1))
    
    #print('useless computations')
    #:todo: This can be improved
    # compute the distance along theta where the segment is crossed
    rmax = d[:,np.newaxis,np.newaxis]/cospsi
    # if the line cannot be cross, therefore the computation can raise some trouble
    # => set it manually to the good value
    rmax[~ind2] = r
    # take the min between the intersection with the segment and the circle
    rmax = np.minimum(r,rmax)
    
    # array containing the evaluation of the function that will be integrated
    R = np.zeros((pos.shape[0],Nth-1,quadt.pts.shape[0], Nr-1,
                  quadr.pts.shape[0],3))

    # intervals for each integral along the radial axis (in r_max unit)
    r_temp = np.linspace(0,1,Nr)
    avr = 0.5*(r_temp[:-1]+r_temp[1:])
    diffr = 0.5*(r_temp[1:]-r_temp[:-1])
    # radius that will be computed
    temp = rmax[...,np.newaxis,np.newaxis]*\
        (diffr[:,np.newaxis]*quadr.pts+avr[:,np.newaxis])
    R[...,0] = pos[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,0]\
               + temp*delta[np.newaxis,...,np.newaxis,np.newaxis,0]
    R[...,1] = pos[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,1]\
               + temp*delta[np.newaxis,...,np.newaxis,np.newaxis,1]
    R[...,2] = pos[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,2]

    # compute the norm of the vector
    R = np.sum(R**2,axis=5)**(-1.5)
    # sum over all the index (theta and r quadrature formula, and, sector)
    # R quadrature
    omega = np.sum(np.sum(temp*R*quadr.w,axis=4)*diffr,axis=3)
    # Theta quadrature
    omega = np.sum(diff*np.sum(rmax*omega*quadt.w,axis=2),axis=1)
    
    # multiply by the scalar product between the position and the normal (to the disk) vector 
    omega *= np.abs(pos[:,2])

    # change the area that we want to compute
    omega[~ind] = solid_angle_disk(pos[~ind,:],r)-omega[~ind]
    temp = solid_angle_disk(pos[~ind,:],r)
        
    return omega





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


def low_pass_box(s,nc):
    """returns the low pass filtered frequency sequence of s, with critical frequency set by location nc. nc must be less than half of the length of s. Assuming s is a frequency domain spectra which obey numpy.fft.fft format.
    ideal box filter is used, which means the frequencies higher than that set by nc will be erased totally, and the frequencies lower than nc will be untouched.
    Inputs:
        s: array_like, frequency spectra that need to be filtered
	nc: int, the critical frequency index above which the signal will be erased
    """

    n = len(s)
    if nc>n/2:
        print("Warning: critical frequency out of input range, nothing will happen to the input spectra.")
        return s
    mask_plus = np.arange(n)<=nc
    mask_minus = np.arange(n)>= n-nc
    s_filtered = np.zeros((n),dtype='complex')
    s_filtered[mask_plus] = s[mask_plus]
    s_filtered[mask_minus] = s[mask_minus]
    return s_filtered	

def high_pass_box(s,nc):
    """returns the high pass filtered frequency sequence of s, with critical frequency set by location nc. nc must be less than half of the length of s. Assuming s is a frequency domain spectra which obey numpy.fft.fft format.
    ideal box filter is used, which means the frequencies lower than that set by nc will be erased totally, and the frequencies higher than nc will be untouched.
    Inputs:
        s: array_like, frequency spectra that need to be filtered
	nc: int, the critical frequency index below which the signal will be erased
    """

    n = len(s)
    if nc>n/2:
        print("Warning: critical frequency out of input range, the whole input spectra will be erased.")
        return np.zeros((n))
    mask_plus = np.arange(n)<=nc
    mask_minus = np.arange(n)>= n-nc
    s_filtered = np.copy(s)
    s_filtered[mask_plus] = 0
    s_filtered[mask_minus] = 0
    return s_filtered	

def band_pass_box(s,nl,nh):
    """A composition of high_pass_box and low_pass_box. nl and nh are lower and higher frequency domain indices respectively, which are in turn passed into low/high_pass_box functions.
    Inputs:
        s: array_like, frequency spectra that need to be filtered
        nl: int, the critical frequency index below which the signal will be erased 
        nh: int, the critical frequency index above which the signal will be erased
    """
    s_low_filtered = high_pass_box(s,nl)
    return low_pass_box(s_low_filtered,nh)


def correlation(s1,s2):
    """Calculate the correlation between two arrays of data. 
    s1 and s2 can be multi-dimensional, the average will be taken over all the dimensions. Returns the correlation, which will be a (complex) number between 0 and 1 (in the sense of the modular).  
    """
    s1_tilde = s1#-np.average(s1)
    s2_tilde = s2#-np.average(s2)
    s = np.average(s1_tilde*np.conj(s2_tilde))
    s1_mod = np.sqrt(np.average(s1_tilde*np.conj(s1_tilde)))
    s2_mod = np.sqrt(np.average(s2_tilde*np.conj(s2_tilde)))
    return s/(s1_mod*s2_mod)


def sweeping_correlation(s1,s2,dt=1,nt_min=100):
    """Calculate the correlation of two given time-series signals.
    
    Correlation is defined as:

    gamma(s1,s2) = <(s1_tilde * conj(s2_tilde))>/sqrt(<|s1_tilde|^2> <|s2_tilde|^2>)

    where s1_tilde = s1 - <s1>, <...> denotes time average.

    Sweeping correlation is carried out by correlating one signal to a delayed(or advanced) version of the other signal.
    
    Arguments:
        s1,s2: signals to be correlated, ndarray with same shape, the first dimension is "time".
        dt: int, sweeping step size, move s2 dt units in time every step, and carryout another correlation with s1
        nt_min: optional, int, the minimum time overlap for sweeping correlation, the average must be taken over longer time period than set by this, otherwise sweeping will stop. Default to be 100. 
        
    Returns:
        SCorrelation: ndarray, same shape as s1 and s2 except for the first dimension, the first dimension length will be total number of sweeping correlations, it's determined by dt, nt_min, and the original time series length. Indexing convention for time dimension is similar to that in fft, if total length is 2n+1, index 0 is for correlation without moving, index 1 to n for s2 delayed compared to s1, index -1 to -n for s2 advanced compared to s1.
        
    """

    assert (s1.shape == s2.shape),'Shapes of two signals don\'t match. s1:{0},s2:{1}'.format(str(s1.shape),str(s2.shape))
    
    shape = s1.shape
    nt = shape[0]
    assert (nt >= nt_min ),'signal length {0} is shorter than minimum length: {1}.'.format(nt,nt_min)
    spatial_shape = shape[1:]
    
    n_wing = int((nt-nt_min)/dt) #length of single wing of the result    
    
    n_sweep = n_wing*2 + 1 #total sweep correlation numbers
    
    SCorrelation = np.empty((n_sweep,)+spatial_shape,dtype = 'complex128') #concatenate last dimension to spatial dimensions
    
    #first get rid of the mean signal
    s1 = s1 - np.mean(s1,axis=0)
    s2 = s2 - np.mean(s2,axis=0)    
    
    for i in range(-n_wing,n_wing+1):
        delta_t = dt*i
        if delta_t < 0:#when s2 is moved advance to s1
            s1_moved = s1[:delta_t,...]
            s2_moved = s2[-delta_t:,...]
            SCorrelation[i] = np.average(s1_moved*np.conj(s2_moved),axis=0)/np.sqrt(np.average(s1_moved*np.conj(s1_moved),axis=0) * np.average(s2_moved*np.conj(s2_moved), axis=0))
        elif delta_t == 0:#when not moved
            SCorrelation[i] = np.average(s1*np.conj(s2),axis=0)/np.sqrt(np.average(s1*np.conj(s1),axis=0) * np.average(s2*np.conj(s2),axis=0))
        else: #when s2 is delayed to s2
            s1_moved = s1[delta_t:,...]
            s2_moved = s2[:-delta_t,...]
            SCorrelation[i] = np.average(s1_moved*np.conj(s2_moved),axis=0)/np.sqrt(np.average(s1_moved*np.conj(s1_moved),axis=0) * np.average(s2_moved*np.conj(s2_moved), axis=0))
    return SCorrelation

