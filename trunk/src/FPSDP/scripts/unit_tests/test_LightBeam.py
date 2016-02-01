# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 18:26:40 2016

@author: lei

test LightBeam
"""
import numpy as np

import FPSDP.Maths.LightBeam as lb
import FPSDP.Geometry.Grid as Grid


grid3d = Grid.Cartesian3D(Xmin = 150, Xmax = 300, NX=201, Ymin=-40, Ymax=40,
                          NY=256, Zmin=-40, Zmax=40, NZ=256)

omega = 8e11

wave_length = 2*np.pi*3e10/omega


gb_test = lb.GaussianBeam(wave_length, 200, 0, 1, tilt_v=np.pi/18)

E_test = gb_test(grid3d.get_ndmesh())

