"""Calculate the recieved light intensity by carrying out the integral along the light path

I(s) = integral(s_0,s)[alpha(s')*exp(-(tau - tau'))*omega^2/(8 pi^3 * c^2) * T(s')] ds' --- ref[1] Eq(2.2.13-2.2.15)

[1] 1983 Nucl. Fusion 23 1153
"""

import .Detector as dtc
import .alpha1 as a1
import numpy as np
import ..Plasma.TestParameter as tp

def Intensity(Dtcs,)