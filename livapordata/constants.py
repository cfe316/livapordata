# some constants for general use with simulations.

# "CODATA recommended values of the
#   fundamental physical constants: 2014"
# Rev. Mod. Phys., Vol. 88, No. 3, July–September 2016
from scipy.constants import physical_constants

from scipy.constants import torr as TORR_TO_PASCALS
from scipy.constants import bar as BARS_TO_PASCALS
from scipy.constants import atm as ATM_TO_PASCALS
from scipy.constants import inch as in_to_m

from scipy.constants import Boltzmann as kB
from scipy.constants import eV

u, _, _ = physical_constants['unified atomic mass unit']

del physical_constants
