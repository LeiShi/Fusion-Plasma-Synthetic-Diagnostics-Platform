# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 15:24:49 2016

@author: lei

This module keeps track of all available synthetic diagnostics on FPSDP.

Information and remaining issues can be added here for future review.

Users should not modify this module.

Developers should add new capabilities or update existing ones when they've
been fully tested and ready to use.
"""
from .Reflectometry import FWR2D, FWR3D
from .ECEI import ECEI1D
from . import BES


Avaliable_Diagnostics = ['FWR2D','FWR3D','ECEI1D','BES']

Entry_Modules = {'FWR2D':FWR2D, 'FWR3D':FWR3D, 'ECEI1D':ECEI1D, 'BES':BES}

