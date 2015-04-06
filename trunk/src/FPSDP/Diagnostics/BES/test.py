import numpy as np


import FPSDP.Diagnostics.BES.bes as bes

profiler = False

if profiler:
    import cProfile, pstats, StringIO
    pr = cProfile.Profile()


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

#b1d1 = bes_.beam
#dl1 = np.sqrt(np.sum((b1d1.get_mesh()-b1d1.get_origin())**2,axis = 1))
if profiler:
    pr.enable()

netilde = bes_.get_bes()

if profiler:
    pr.disable()


print netilde
foc = bes_.pos_foc

if profiler:
    s = StringIO.StringIO()
    sortby = 'time'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()


np.savez('data/test2',netilde,foc)


