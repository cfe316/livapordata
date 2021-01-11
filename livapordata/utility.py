from numpy import sqrt, pi
from livapordata.constants import kB


def pickle_obj(obj, filename):
    output = open(filename, 'wb')
    pickle.dump(obj, output)
    output.close()


#-------------------------------------


def error_bands(value, percent):
    e_plus = (1 + 0.01 * percent) * value
    e_minus = (1 - 0.01 * percent) * value
    return e_minus, e_plus


# ---------------------------------
# physics formulas


def langmuir_flux(density, t_kelvin, mass):
    """Calculates langmuir flux of particles # / m^2 s

    Parameters
    ----------
    density, m⁻³

    t_kelvin, K

    mass, kg

    Returns
    -------
    flux, m⁻² s⁻¹

    Notes
    -----
    .. math::

         \Gamma = n \sqrt{\frac{k_B T_k}{2 \pi m}}

    """
    flux = density * sqrt(kB * t_kelvin / (2 * pi * mass))
    return flux


# ---------------------------------
