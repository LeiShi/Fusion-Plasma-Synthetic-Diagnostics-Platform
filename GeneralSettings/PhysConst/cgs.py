# Define all the physical constants in cgs

# module depends on math constants
import math 

# quantities' values based on NRL Plasma Formulary Revised 2009

# elementary charge
e = 4.8032e-10

# electron mass
m_e = 9.1094e-28

# proton mass
m_p = 1.6726e-24

# speed of light
c = 2.9979e10

# permittivity of free space (not used)
# eps_0 = 1

# permeability of free space (not used)
# mu_0 = 1

# more constants can be added later...

# function that tells the name of the unit system
def tell():
    return 'cgs'