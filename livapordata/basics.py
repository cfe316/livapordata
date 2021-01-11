from livapordata.constants import u, eV

#Atomic weights of the elements 2013 (IUPAC Technical Report)
mLiConventionalAtomicWeight = 6.94
mLi = mLiConventionalAtomicWeight * u

def surface_tension(t_k=453.7):
    """Surface tension as a function of temperature

    Parameters
    ----------
    t_k : float, default=453.7
        K, temperature. Default is the melting point.

    Returns
    -------
    σ : float
        N m, surface tension

    Davison, H.W.
    "Complication of Thermophysical Properties of Liquid Lithium."
    NASA Technical Note TN D-4650.
    Cleveland, Ohio: Lewis Research Center, July 1968.

    "Using equation [below], the data are correlated with
     a standard deviation of +- 1.9 percent. The maximum difference
     between the data and the correlation is +5.2 percent."
    """
    σ = 0.447 - 1.07e-4 * t_k - 1.351e-8 * t_k**2
    return σ


# rough estimate for liquid density
rho_liquid = 500  # kg m⁻³

# estimate for solid density
rho_solid = 534  # kg m⁻³

T_melting = 180.50  # CRC handbook

# Via wikipedia:
#     "
#     J.A. Dean (ed.), Lange's Handbook of Chemistry (15th Edition),
#     McGraw-Hill, 1999; Section 6, Thermodynamic Properties;
#     Table 6.4, Heats of Fusion, Vaporization, and Sublimation
#     and Specific Heat at Various Temperatures of
#     the Elements and Inorganic Compounds
#
#     Values refer to the enthalpy change in the conversion of liquid to gas at
#     the boiling point (normal, 101.325 kPa).
#     "
latent_heat_vaporization_kJ_per_mol = 147.1  # kJ / mol
latent_heat_vaporization_J_per_kg = (
    1e3 * latent_heat_vaporization_kJ_per_mol) / (
        1e-3 * mLiConventionalAtomicWeight)
