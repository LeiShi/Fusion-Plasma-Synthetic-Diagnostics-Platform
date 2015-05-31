import numpy as np


import FPSDP.Diagnostics.BES.bes as bes

profiler = False
parallel = False

print 'Parallel: ', parallel

if profiler:
    import cProfile, pstats, StringIO
    pr = cProfile.Profile()

name = 'FPSDP/Diagnostics/BES/bes.in'
f = open(name,'r')
input_ = f.read()
f.close()
bes_ = bes.BES(name,parallel)



if profiler:
    pr.enable()

I = bes_.get_bes()

if profiler:
    pr.disable()


print I
foc = bes_.pos_foc

if profiler:
    s = StringIO.StringIO()
    sortby = 'time'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()

psin = bes_.get_psin(foc)

np.savez('data/test',I,psin,foc,input_)


