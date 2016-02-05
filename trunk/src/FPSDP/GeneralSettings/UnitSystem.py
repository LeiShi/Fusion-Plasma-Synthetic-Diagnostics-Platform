"""Unit Systems defined in this module
"""

#module depends on the built-in library 'math'
import math

class UnitError(Exception):
    """Unit related Exception class.
    """
    def __init__(self, *a):
        self.args = a


class UnitSystem:
    """Base class for unit systems

    Attributes:
        :var _ConstDic: dictionary contains all the useful physical constants.
    
    Methods:    
        :py:method:`list_const`: returns the _ConstDic dictionary.
        :py:method:`add_const`: add new const into the _ConstDic.
        :py:method:`del_const`: remove existing const from _ConstDic.
        :py:method:`rename_const`: change the key for a const to obey user 
                                   defined synbol rules.
    """
    def __init__(self, name, description, **CD):
        """Initialize with a name and/or a keyword argument list containing the
        constants
        """
        self._name = name
        self._description = description
        self._ConstDic = dict(**CD)

    def __getitem__(self,cname):
        """get consts by the name
        """
        try:
            return self._ConstDic[cname]
        except KeyError:
            print 'const "{0}" not defined in unit "{1}"'.format(cname,
                                                                 self._name)
            raise
        except:
            raise
            
    def __str__(self):
        return '{}\n{}\n'.format(self._name, self._description)

    def list_const(self):
        """ returns the dictionary contains all the (name,value) pairs of the 
        consts
        """
        return self._ConstDic

    def add_const(self, **C):
        """Add one or more constants using a keyword argument list.

        :raise UnitError: If no argument is passed, a UnitError exception is 
                          raised.
        
        :raise UnitError: If one of the passed keys already exists, the value 
                          passed will be compared with the value stored. If the
                          relative difference is less than 10e-4, the value 
                          will be **CHANGED** to the new value. However, if the 
                          difference is larger than that, a UnitError exception
                          will be raised.
        """
        if not C:
            raise UnitError("Keyword arguments needed for adding new constants\
 to the unit system.")
        else:
            for key in C.keys():
                if key in self._ConstDic.keys():
                    dif = ( float(C[key])-self._ConstDic[key] ) / C[key]
                    if dif <= 1e-4:
                        self._ConstDic[key] = C[key]
                        continue
                    else:
                        raise UnitError("Const value conflict! Please double \
check the unit system you are using!\nPossibly a name conflict, check the \
names as well.")
                else:
                    self._ConstDic[key] = C[key]

    def del_const(self, *names):
        """Delete the consts for given names.

        If no name is given, a UnitError exception will be raised.
        If a given name doesn't exist, a UnitError exception will be raised.
        """
        if not names:
            raise UnitError("No names passed for deletion.")
        else:
            for n in names:
                if not ( n in self._ConstDic.keys() ):
                    raise UnitError("name {0} not found for deletion.".\
                                     format(n))
                else:
                    del self._ConstDic[n]

    def rename_const(self, **NameDic):
        """Change the name of a exsiting constant.

        Keyword arguments are accepted with the key being the EXISTING name 
        and the value being the new name.
        
        :raise UnitError: If keywords contain non-existing names, a UnitError 
                          exception will be raised.
        """
        for oldname in NameDic.keys():
            if not (oldname in self._ConstDic.keys() ):
                raise UnitError("Const {0} dosen't exist, can not rename it".format(oldname))
            else:
                self._ConstDic[ NameDic[oldname] ] = self._ConstDic[oldname]
                del self._ConstDic[oldname]

# some pre-defined unit systems
# quantities' values based on NRL Plasma Formulary Revised 2009
cgs = UnitSystem('cgs','Length: centi-meter\nMass: gram\nTime: second',
                 
# elementary charge
                 e = 4.8032e-10,
# electron mass
                 m_e = 9.1094e-28,
# proton mass
                 m_p = 1.6726e-24,
# speed of light
                 c = 2.9979e10,
# energy associated with 1keV in erg
                 keV = 1.6022e-9
# permittivity of free space (not used)
#                eps_0 = 1,
# permeability of free space (not used)
#                mu_0 = 1,
# more constants can be added later...
                 )

SI = UnitSystem('SI','Length: meter\nMass: kilo-gram\nTime: second',
# elementary charge
                e = 1.6022e-19,
# electron mass
                m_e = 9.1094e-31,
# proton mass
                m_p = 1.6726e-27,
# speed of light
                c = 2.9979e8,
# permittivity of free space
                eps_0 = 8.8542e-12,
# permeability of free space
                mu_0 = 4e-7 * math.pi,
# energy associated with 1keV
                keV = 1.6022e-16,
# Planck constant * speed of light
                hc = 1.986445e-25,
#atomic mass unit
                amu = 1.660538e-27
# more constants can be added later...
)
