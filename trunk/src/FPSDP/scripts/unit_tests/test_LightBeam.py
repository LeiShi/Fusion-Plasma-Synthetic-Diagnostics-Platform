# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 18:26:40 2016

@author: lei

test LightBeam
"""

import FPSDP.Maths.LightBeam as lb
import FPSDP.Geometry.Grid as Grid

grid3d = Grid.Cartesian3D(Xmin = -100, Xmax = 100, NX=201, Ymin=-20, Ymax=20,
                          NY=41, Zmin=-20, Zmax=20, NZ=41)

gb = lb.GaussianBeam(1, 0, 0, 10)

E = gb(grid3d.get_ndmesh())

