import numpy as np
from scipy import polyfit
from lithdata.constants import kB

def reference_diameter_VHS(mass, eta_ref, T_ref, omega):
    """
    Calculates the reference diameter for a VHS model so that
        eta(T_ref) == eta_ref

    Parameters:
        mass: in kilograms
        eta_ref: viscosity at the reference temperature, in Pa s
        T_ref: reference temperature, in Kelvin
        omega: Unitless. Previous best fit to viscosity: eta ~ T^{omega}

    Returns:
        Reference diameter in meters.

    Reference:
        Bird, G. A., The DSMC Method, Version 1.2, 2013.
        Chapter 2, Equation (43)
    Formula:
        Note that the reference uses $\mu$ for viscosity; here we use $\eta$.
        $$d_{\mathrm{ref}} = \left(\frac{15(m k_B T_{\mathrm{ref}} / \pi)^{1/2}}{2 (5 - 2 \omega)(7 - 2 \omega) \eta_{\mathrm{ref}}}\right)^{1/2}$$

    """
    numerator = 15 * (mass * kB * T_ref / np.pi) ** (1 / 2)
    denominator = 2 * (5 - 2 * omega) * (7 - 2 * omega) * eta_ref
    d_ref = (numerator / denominator) ** (1 / 2)
    return d_ref


def VHS_model_from_viscosity(visc_func, mass, T_min, T_max, T_ref=0):
    """
    Calculates the reference diameter for a VHS model.

    Parameters:
        visc_func: a function of one variable, temperature in Kelvin,
            returns viscosity at that temperature in Pa s.
        mass: mass of the particle, in kilograms
        T_min, T_max: range over which to perform the fit.
        T_ref: reference temperature, in Kelvin. 
            Default is the mean of T_min and T_max.
    Returns:
        A dictionary:
            mass : in kilograms
            T_ref: in Kelvin
            omega: Unitless. Best fit to viscosity: eta ~ T^{omega}
            d_ref: Best fit for the reference diameter, in meters.
                Note that because this is a best fit, the fit VHS model's
                viscosity at the reference temperature is likely different
                from that of the supplied viscosity function.
    """
    NUM_POINTS_TO_FIT_OVER = 100 # bignum
    if T_ref == 0:
        T_ref = (T_min + T_max) / 2
    TK = np.linspace(T_min, T_max, NUM_POINTS_TO_FIT_OVER)
    eta = visc_func(TK)
    # fit in log - logspace
    x = np.log(TK)
    y = np.log(eta)
    omega, b = polyfit(x, y, deg=1)

    # Calculate d_ref 
    A_best = np.exp(b) # eta = A_best T^omega
    numerator = 15 * np.sqrt(mass * kB * T_ref / np.pi)
    denominator = 2 * A_best * (7 - 2 * omega) * (5 - 2 * omega) * T_ref ** omega
    d_best = (numerator / denominator)**(1/2)
    return {'d_ref': d_best, 'omega': omega, 'T_ref': T_ref, 'mass': mass}
