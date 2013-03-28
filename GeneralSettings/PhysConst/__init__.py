import cgs
import SI

# list of currently defined unit systems
ValidSys = ('cgs','SI')


# default Unit System set to be cgs
CurUnitSys = cgs



def set_unit(us):
    """
    set the unit system to be used.

    us := string contains the name of a valid unit system 
    """
    if us == 'cgs' or us == 'CGS':
        CurUnitSys = cgs
    elif us == 'SI' or us == 'si':
        print 'unit set to SI'
        CurUnitSys = SI
        print CurUnitSys.tell()
    else:
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

