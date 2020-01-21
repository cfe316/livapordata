from lithdata.constants import u

#Atomic weights of the elements 2013 (IUPAC Technical Report)
mLiConventionalAtomicWeight = 6.94 
mLi = mLiConventionalAtomicWeight * u

# estimate for lithium surface tension

def surface_tension(TK=453.7):
    """Surface tension as a function of temperature

    Parameters:
        Temperature in Kelvins. Default: melting point.
    Returns:
        Surface tension in Newton meters
    
    Davison, H.W. 
    "Complication of Thermophysical Properties of Liquid Lithium."
    NASA Technical Note TN D-4650. 
    Cleveland, Ohio: Lewis Research Center, July 1968.

    "Using equation [below], the data are correlated with
     a standard deviation of +- 1.9 percent. The maximum difference
     between the data and the correlation is +5.2 percent."
    """
    sigma = 0.447 - 1.07e-4 * TK - 1.351e-8 * TK ** 2
    return sigma

# rough estimate for liquid density
rho_liquid = 500 # kg m⁻³

# estimate for solid density
rho_solid = 534 # kg m⁻³
