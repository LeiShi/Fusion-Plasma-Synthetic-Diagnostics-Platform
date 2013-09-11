#!/usr/bin/env python
from test import cprint,set,pyprint,arraysum,arrayadd
import numpy as np

n = 'Pu Ying'
a = 25

cprint()
set(name = n, age = a)
cprint()

set(name = "nobody",age = -1)
cprint()

pyprint('just try pyprint')


a1 = np.linspace(1.0,10.0,10)
a2 = np.linspace(2.0,11.0,20)
b1 = a1.reshape((2,5))
b2 = a2.reshape((2,2,5))
b3 = np.linspace(2.,10.,9).reshape((3,3))

print b1
print b2
print b3

#c = arrayadd(b1,b2)
c_py = b1+b2
#print c
print c_py

#c_wrong = arrayadd(b1,b3)
#c_wrong_py = b1+b3
#print c_wrong_py
