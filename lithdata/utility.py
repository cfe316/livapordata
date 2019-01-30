from numpy import sqrt, pi
from lithdata.constants import kB

def pickle_obj(obj, filename):
    output = open(filename, 'wb')
    pickle.dump(obj, output)
    output.close()

# ---------------------------------
# physics formulas

def langmuir_flux(density, t_kelvin, mass):
    """Calculates langmuir flux of particles # / m^2 s
    density in #/m^3
    t_kelvin is temperature in kelvin
    mass in kg    
    """
    flux = density * sqrt(kB * t_kelvin / (2 * pi * mass))
    return flux

# ---------------------------------
