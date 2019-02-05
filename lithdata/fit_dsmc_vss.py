import numpy as np
from scipy import polyfit
from lithdata.constants import kB, mLi

def schmidt_number_from_VSS(vss):
    a = vss['alpha']
    w = vss['omega']
    return 5 * (2 + a) / (3 * a * (7 - 2 * w))

def schmidt_number(TK, eta, D11, p0, mass):
    Sc = eta / (D11 * p0 * mass / (TK * kB))
    return Sc

def reference_diameter_VSS(mass, eta_ref, T_ref, omega, alpha):
    """
    Calculates the reference diameter for a VSS model so that
        eta(T_ref) == eta_ref

    Parameters:
        mass: in kilograms
        eta_ref: viscosity at the reference temperature, in Pa s
        T_ref: reference temperature, in Kelvin
        omega: Unitless. Previous best fit to viscosity: eta ~ T^{omega}
        alpha: Unitless. Determined from fit to Schmidt number.

    Returns:
        Reference diameter in meters.

    Reference:
        Bird, G. A., The DSMC Method, Version 1.2, 2013.
    Formula:
        Note that the reference uses $\mu$ for viscosity; here we use $\eta$.

    """
    numerator = 5 * (alpha + 1 ) * (alpha + 2) * (mass * kB * T_ref / np.pi) ** (1 / 2)
    denominator = 4 * alpha * (5 - 2 * omega) * (7 - 2 * omega) * eta_ref
    d_ref = (numerator / denominator) ** (1 / 2)
    return d_ref


def VSS_model_from_eta_and_D11(visc_func, diff_func, mass, T_min, T_max, T_ref=0):
    """
    Calculates a VSS model.

    Parameters:
        visc_func: a function of one variable, temperature in Kelvin,
            returns viscosity at that temperature in Pa s.
        diff_func: a function of one variable, temperature in Kelvin,
            returns diffusion coefficient at the reference pressure of 1e5 Pa.
        mass: mass of the particle, in kilograms
        T_min, T_max: range over which to perform the fit.
        T_ref: reference temperature, in Kelvin. 
            Default is the mean of T_min and T_max.
    Returns:
        A dictionary:
            mass : in kilograms
            T_ref: in Kelvin
            omega: Unitless. Best fit to viscosity: eta ~ T^{omega}
            alpha: Unitless. Best fit to Schmidt number.
            d_ref: Best fit for the reference diameter, in meters.
                Note that because this is a best fit, the fit VSS model's
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

    # calculate Schmidt numbers and alpha
    D11 = diff_func(TK)
    Sc = schmidt_number(TK, eta, D11, 1e5, mLi)
    alpha= 10 / (Sc * (21 - 6 * omega) - 5)
    best_fit_alpha, = polyfit(TK, alpha, deg=0)
    a = best_fit_alpha # also equals the mean for a deg=0 polynomial ;)

    # Calculate d_ref 
    A_best = np.exp(b) # eta = A_best T^omega
    numerator = 5 * np.sqrt(mass * kB * T_ref / np.pi) * (1 + a) * (2 + a)
    denominator = 4 * a * A_best * (7 - 2 * omega) * (5 - 2 * omega) * T_ref ** omega
    d_best = (numerator / denominator)**(1/2)
    return {'d_ref': d_best, 'omega': omega, 
            'T_ref': T_ref, 'mass': mass, 'alpha': a}
