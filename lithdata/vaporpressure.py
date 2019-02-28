import numpy as np
from numpy import sqrt, log, exp, pi
from lithdata.constants import kB

### Vapor pressure
def press_Browning_and_Potter(t_kelvin):
    """Lithium vapor pressure in Pascals

    Browning, P, and P. E. Potter. “Assessment of the Experimentally 
    Determined Vapour Pressures of the Liquid Alkali Metals.”
    In Handbook of Thermodynamic and Transport Properties of Alakali Metals, 
    349–58. Oxford: Blackwell Scientific Publications, 1985.

    Section 6.2, Page 350, Equation (2)
    Valid over 1057 K < T < 2156 K.
    """
    c1 = 13.0719
    c2 = -18880.659
    c3 = -0.4942
    megapascals_to_pascals = 1e6
    p_megapascals = exp(c1 + c2 / t_kelvin + c3 * log(t_kelvin))
    pressure_pa = p_megapascals * megapascals_to_pascals
    return pressure_pa


