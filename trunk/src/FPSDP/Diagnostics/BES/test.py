import numpy as np


import FPSDP.Diagnostics.BES.bes as bes

profiler = False
parallel = False

print 'Parallel: ', parallel

if profiler:
    import cProfile, pstats, StringIO
    pr = cProfile.Profile()


bes_ = bes.BES('FPSDP/Diagnostics/BES/bes.in',parallel)

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


np.savez('data/test',netilde,foc)


