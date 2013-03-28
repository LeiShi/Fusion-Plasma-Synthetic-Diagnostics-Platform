# Define all the physical constants in SI

# module depends on math constants
import math 

# quantities' values based on NRL Plasma Formulary Revised 2009

# elementary charge
e = 1.6022e-19

# electron mass
m_e = 9.1094e-31

# proton mass
m_p = 1.6726e-27

# speed of light
c = 2.9979e8

# permittivity of free space
eps_0 = 8.8542e-12

# permeability of free space
mu_0 = 4e-7 * math.pi

# more constants can be added later...

# function that tells the name of the unit system
def tell():
    return 'SI'