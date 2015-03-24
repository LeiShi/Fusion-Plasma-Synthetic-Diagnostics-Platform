import scipy.interpolate as ip
import matplotlib.pyplot as mp
import numpy as np
import math


import FPSDP.Diagnostics.Beam.beam as be
import FPSDP.Diagnostics.BES.bes as bes
import FPSDP.Plasma.XGC_Profile.load_XGC_profile as xgc
import FPSDP.Geometry.Grid as Grid
import scipy.io as sio

"""import FPSDP.Maths.Integration as integ

def f(r,th):
    return r*np.sqrt(4.0-r**2)

quad = integ.integration_points(1,'GL5')
theta = np.linspace(0,2*np.pi,100)

av = 0.5*(theta[:-1] + theta[1:])
diff = 0.5*np.diff(theta)
rmax = 2.0
pts = quad.pts
rpts = 0.5*rmax*(pts + 1.0)
I = 0.0
for i in range(len(av)):
    th = av[i]+diff[i]*pts
    R, T = np.meshgrid(rpts,th)
    I_r = np.einsum('i,ij->j',quad.w,f(R,T))
    I_th = 0.5*rmax*diff[i]*np.sum(quad.w*I_r)
    I += I_th
print I
"""
bes_ = bes.BES('FPSDP/Diagnostics/BES/bes.in')
xgc_ = bes_.beam.data

b1d1 = bes_.beam
dl1 = np.sqrt(np.sum((b1d1.get_mesh()-b1d1.get_origin())**2,axis = 1))
print bes_.intensity(range(2),0)

mp.figure()
ms = b1d1.get_mesh()
foc = bes_.fib_pos[0,:] + bes_.focal*bes_.op_direc
mp.plot(ms[:,0],ms[:,1])
mp.plot(foc[0],foc[1],'o',markersize=4)

mp.figure()
mp.plot(dl1,b1d1.density_beam[0,0,:])

#np.save('beam_eq',b1d1.density_beam[0,0,:])
be = np.load('beam_eq.npy')
print 'change condition eq'
mp.plot(dl1,be,'r', label='eq')


mp.figure()
mp.plot(dl1,(b1d1.density_beam[0,0,:]-be)/be)

mp.show()

