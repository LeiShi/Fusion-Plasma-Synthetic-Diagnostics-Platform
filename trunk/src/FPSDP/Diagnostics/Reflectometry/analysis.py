"""General analysis functions for reflectometry related data

Contains:
    phase(raw_sig) : processes the raw signal's phase, creates a new phase series such that the phase change in one time step is not larger than PI. In this way, the phase jump across -PI and PI boundary is avoided. The phase curve is more smooth and the range is also extended to (-inf,inf). 
"""

import numpy as np

def phase(raw_sig):
    """Calculate the extended phase curve for a given complex signal array.

    The purpose of extending the phase range to (-infi,+infi) is to avoid jumps from +pi -> -pi or the other way around on the normal [-pi,pi) range. In this case, the phase curve looks much smoother and more meaningful.
    The method we are using is first calculate the phase for each time step in the normal [-pi,pi) range, then, calculate the phase change for each time interval : dphi. For dphi>pi, we pick dphi-2*pi as the new phase change; and for dphi < -pi, we pick dphi+2*pi. In other words, we minimize the absolute value of the phase change. This treatment is valid if time step is small compared to plasma changing time scale, so the change of reflected phase shouldn't be very large.

    Arguments:
        raw_sig: 1d complex array, the complex signal gotten from either measurement or synthetic reflectometry runs.
    Return:
        (new_phase,new_dph) tuple: new_phase is the modified phase series, new_dph is the phase change series.
    """
    phase_raw = np.angle(raw_sig) # numpy.angle function gives the angle of a complex number in range[-pi,pi)

    dph = phase_raw[1:]-phase_raw[0:-1] #the phase change is defined on each time intervals, so the total length will be 1 shorter than the phase array.

    dph_ext = np.array([dph-2*np.pi,dph,dph+2*np.pi]) #intermediate array that contains all 3 posibilities of the phase change

    dph_arg = np.argmin(np.abs(dph_ext),axis = 0) #numpy.argmin function pick out the index of the first occurance of the minimun value in the array along one chosen axis. Since the axis 0 in our array has just 3 elements, the dph_arg will contain only 0,1,2's.

    new_dph = dph + (dph_arg-1)*2*np.pi # notice that in dph_arg, 0 corresponds dph-2*pi being the chosen one, 1 -> dph, and 2 -> dph+2*pi, therefore, this expression is valid for all 3 cases.

    new_phase = np.zeros_like(phase_raw)+phase_raw[0]
    new_phase[1:] += new_dph.cumsum() # numpy.ndarray.cumsum method returns the accumulated array, since we are accumulating the whole dph_new array, the phase we got is relative to the initial phase at the start of the experiment.

    return (new_phase,new_dph)
