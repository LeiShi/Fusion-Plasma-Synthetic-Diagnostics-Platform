#!/bin/python

#recursively read netcdf files for plasma profile and FWR2D output
#create the data input file for VisIt to make the movie 

import FPSDP.Visualization.VisIt as vi
import numpy as np
import os

fluc_file_head = 'fluctuation'
Tstart = 699
Tend =700
Tstep = 5
reflect_file = 'schradi.cdf'
para_out_file_head = 'para_out'
full_wave_out_file_head = 'fullw_out'

run_dir = '../runs/'
vi_out_dir = '../vtk_files/'

wavefreq = 73 # in GHz

for i in range(Tstart,Tend+1,1):
    flucfname = run_dir+str(i)+'/'+fluc_file_head+str(i)+'.cdf'
    reffname = run_dir+str(i)+'/'+str(wavefreq)+'/'+reflect_file
    flucoutfname = vi_out_dir + fluc_file_head+str(i)+'.vtk'
    paraoutfname = vi_out_dir + para_out_file_head + str(i) + '.vtk'
    fullwoutfname = vi_out_dir + full_wave_out_file_head +str(i) + '.vtk'
    
    fwr = vi.FWR_Loader(freq = wavefreq*1E9, flucfname = flucfname, fwrfname = reffname, mode = 'X')

    flucmesh = fwr.load_profile()
    flucmesh.output_vtk(fname = flucoutfname)
    del flucmesh
    
    para_mesh = fwr.load_paraxial()
    para_mesh.output_vtk(fname = paraoutfname)
    del para_mesh
    
    fullw_mesh = fwr.load_fullwave()
    fullw_mesh.output_vtk(fname = fullwoutfname)
    del fullw_mesh
    del fwr
    
        
    
    
