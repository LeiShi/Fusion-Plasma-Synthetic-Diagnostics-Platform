#!/bin/python

#Read parameters from RunScript.spt
#recursively read netcdf files for plasma profile and FWR2D output
#create the data input file for VisIt to make the movie 

import FPSDP.Visualization.VisIt as vi
import numpy as np
import os

fluc_file_head = 'fluctuation'
Tstart = 1
Tend =700
Tstep = 1
reflect_file = 'schradi.cdf'
reflect_out_file_head = 'FWRout'

run_dir = '../runs/'
vi_out_dir = '../VisIt/'

wavefreq = 73 # in GHz

for i in range(Tstart,Tend+1,1):
    flucfname = run_dir+str(i)+'/'+fluc_file_head+str(i)+'.cdf'
    reffname = run_dir+str(i)+'/'+str(wavefreq)+'/'+reflect_file
    flucoutfname = vi_out_dir + fluc_file_head+str(i)+'.vtk'
    refoutfname = vi_out_dir + reflect_out_file_head + str(i) + '.vtk'

    flucmesh = vi.load_profile_from_netcdf_fluctuation(flucfname)
    refmesh = vi.load_reflect_from_netcdf(reffname)
    
    flucmesh.output_vtk(fname = flucoutfname)
    refmesh.output_vtk(fname = refoutfname)
    
        
    
    