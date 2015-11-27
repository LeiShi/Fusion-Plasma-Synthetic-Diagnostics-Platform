# used for saving the data
import numpy as np
# import the diagnostic
import FPSDP.Diagnostics.BES.bes as bes


# choose if between the serial and the multiprocessing code (only the shared memory case is done)
multiprocessing = False
# print the choice betweem parallel or not
print 'Parallel code: ', multiprocessing

# initialize the diagnostic by taking the config file bes.in
# the initialization consist to compute the beam density,
# the geometry, loading a few datas from XGC1, ...
name = './bes.in'

# save the input file in the data
f = open(name,'r')
input_ = f.read()
f.close()

bes_ = bes.BES(name,multiprocessing)
# compute the intensity (number of photons) received by each fiber
# and their psi_n value
I = bes_.get_bes()

# take the position of the focus points
foc = bes_.pos_foc

# compute the psi_n value of each fiber
psi_n = bes_.get_psin(foc)
# save all the datas
np.savez('.//test',I,psi_n,foc,input_)
