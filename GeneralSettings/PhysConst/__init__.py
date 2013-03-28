import cgs
import SI

# list of currently defined unit systems
ValidSys = ('cgs','SI')


# default Unit System set to be cgs
# note that the CurUnitSys variable is set to be a list such that users can modify it later.
# So to reference the Unit object, one needs actually to use CurUnitSys[0]. 
CurUnitSys = [cgs]



def set_unit(us):
    """
    set the unit system to be used.

    us := string contains the name of a valid unit system 
    """
    if us == 'cgs' or us == 'CGS':
        CurUnitSys[0] = cgs
    elif us == 'SI' or us == 'si':
        CurUnitSys[0] = SI
    else:
        #exception handling may be needed here
        print 'Unit system: "'+str(us)+'" not recognised.\nUnit system not set.'
        valid_systems()

def valid_systems():
    """
    print out the valid unit system names.
    """
    print 'Currently valid unit systems are:'
    for us in ValidSys:
        print us
    print 'For detailed information, please use "ToBeSpecified."'

