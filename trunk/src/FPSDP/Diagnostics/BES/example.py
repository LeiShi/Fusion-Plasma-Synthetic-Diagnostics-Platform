# used for saving the data
import numpy as np
# import the diagnostic
import FPSDP.Diagnostics.BES.bes as bes


# choose if between the serial and the parallel code (only the shared memory case is done)
parallel = False
# print the choice betweem parallel or not
print 'Parallel: ', parallel

# initialize the diagnostic by taking the config file bes.in
# the initialization consist to compute the beam density,
# the geometry, loading a few datas from XGC1, ...
bes_ = bes.BES('FPSDP/Diagnostics/BES/bes.in',parallel)

# compute the intensity (number of photons) received by each fiber
netilde = bes_.get_bes()

# take the position of the focus points
foc = bes_.pos_foc

# save all the datas
np.savez('data/test',netilde,foc)
