from livapordata.vaporpressure import press_best

import numpy as np
from numpy import sqrt, log, exp, pi
from scipy.interpolate import interp1d
from scipy.constants import gas_constant as R_gas


def eta1_Vargaftik_and_Yargin(TK):
    """Dynamic viscosity of Li monomers

    Dynamic viscosity of the monomers as a function of temperature.

    Parameters
    ----------
    TK
        K, temperature

    Returns
    -------
    η, Pa s

    References
    ----------
    Vargaftik, N B, and V S Yargin.
    Ch 7.4: Thermal Conductivity and Viscosity of the Gaseous Phase.
    Handbook of Thermodynamic and Transport Properties of Alkali Metals,
    edited by R. W. Ohse, 45. Blackwell Scientific Publications, 1985.

    Equation (56), page 821."""
    eta1 = 1e-7 * (130.6 + 0.1014 * (TK - 1000) - 4.55e-6 * (TK - 1000)**2)
    return eta1


def eta1_Vargaftik_and_Yargin_error(TK):
    """Estimated error in dynamic viscosity

    Parameters
    ----------
    TK
        K, temperature

    Returns
    -------
    Error in η, Pa s

    References
    ----------
    Vargaftik, N B, and V S Yargin.
    Ch 7.4: Thermal Conductivity and Viscosity of the Gaseous Phase.
    Handbook of Thermodynamic and Transport Properties of Alkali Metals,
    edited by R. W. Ohse, 45. Blackwell Scientific Publications, 1985.

    Notes
    -----
    "Errors in the viscosity and thermal conductivity factors for
    the lithium vapour atomic component, due to inaccuracy in
    calculating atom collision integrals, are equal on average
    to 3%, falling from 3.8% to 1.5% with increase of the temperature
    from 700 to 2500 K. The portion of the error which is determined
    by inaccuracy in establishing the value of $\beta^2_{12}$ is
    changed with the concentration of the molecular component, reaching
    its maximum at the saturation line. In the case of viscosity it is
    1 - 6 %, and for thermal conductivity it is 4 - 8 % (for T <= 2000 K)"
    """
    x1, y1 = (700, 3.8)
    x2, y2 = (2500, 1.5)
    return y1 + (y2 - y1) * (TK - x1) / (x2 - x1)


def eta_Vargaftik_and_Yargin(x2, TK):
    """Viscosity with two components, monomer and dimer.

    Parameters
    ----------
    x2 : float
        Mole fraction of the dimer (molecular) component, Li₂

    TK : float
        K, temperature

    Returns
    -------
    η, Pa s

    References
    ----------
    Vargaftik, N B, and V S Yargin.
    Ch 7.4: Thermal Conductivity and Viscosity of the Gaseous Phase.
    Handbook of Thermodynamic and Transport Properties of Alkali Metals,
    edited by R. W. Ohse, 45. Blackwell Scientific Publications, 1985.

    Equation (55), page 821.

    .. math::
        \eta(x_2, T) = \eta_1(T) \left(1 - 3.65 x_2
                                 + 12.5 x_2^2 - 42 x_2^3
                                 + 142 x_2^4 - 479 x_2^5
                                 + 1600 x_2^6\right)$$
    """
    eta1 = eta1_Vargaftik_and_Yargin(TK)
    eta = eta1 * (1 - 3.65 * x2    + 12.5 * x2**2
                      - 42 * x2**3 +  142 * x2**4
                     - 479 * x2**5 + 1600 * x2**6)
    return eta


def eta_sat_Vargaftik_and_Yargin_Table():
    """Dynamic viscosity of saturated Li vapor

    Returns
    -------
    array
        Array of temperature and dynamic viscosity data

    References
    ----------
    Vargaftik, N B, and V S Yargin.
    Ch 7.4: Thermal Conductivity and Viscosity of the Gaseous Phase.
    Handbook of Thermodynamic and Transport Properties of Alkali Metals,
    edited by R. W. Ohse, 45. Blackwell Scientific Publications, 1985.
    Table 36
    """
    t = [700, 725, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975,
        1000, 1025, 1050, 1075, 1100, 1125, 1150, 1175, 1200,
        1225, 1250, 1275, 1300, 1325, 1350, 1375, 1400, 1425,
        1450, 1475, 1500, 1525, 1550, 1575, 1600, 1625, 1650,
        1675, 1700, 1725, 1750, 1775, 1800, 1825, 1850, 1875,
        1900, 1925, 1950, 1975, 2000]
    eta_sats = np.array([98.6, 100.9, 103.0, 105.0, 107.0, 108.9,
            110.6, 112.3, 113.8, 115.3, 116.6, 117.8, 119.0,
            120.0, 121.0, 121.9, 122.7, 123.4, 124.1, 124.7,
            125.3, 125.8, 126.2, 126.7, 127.1, 127.4, 127.8,
            128.1, 128.4, 128.7, 129.0, 129.3, 129.6, 129.9,
            130.2, 130.5, 130.7, 131.0, 131.3, 131.7, 132.0,
            132.3, 132.6, 133.0, 133.3, 133.7, 134.1, 134.4,
            134.8, 135.2, 135.6, 136.0, 136.4])
    data = np.array([t, 1.0e-7 * eta_sats]).T
    return data

def x2_concentration_Vargaftik_and_Yargin(P_kpa, Keq):
    """Molar fraction of the dimer Li₂

    Parameters
    ----------
    P_kpa : float
        kPa, pressure

    Keq: float
        Constant of the chemical equilibrium

    Returns
    -------
    x2 : float
       Molar fraction of the dimer

    References
    ----------
    Equation (80)
    Vargaftik, N B, and V S Yargin.
    Ch 7.4: Thermal Conductivity and Viscosity of the Gaseous Phase.
    Handbook of Thermodynamic and Transport Properties of Alkali Metals,
    edited by R. W. Ohse, 45. Blackwell Scientific Publications, 1985.

    The formula is noted as being from

    L.D. Volyak, Physical Constants and Properties of Substances,
    Thermophysical Properties of Substances 49, GSSSD, Moscow, 1968
    """
    x2 = 1 - 2 / (1 + sqrt(1 + 3.9477e-2 * P_kpa / Keq))
    return x2


def phi_V_and_Y(component, T):
    """Reduced thermodynamical potential

    Parameters
    ----------
    component : int
        1 for the monomer, 2 for the dimer

    T : float
        K, temperature

    Returns
    -------
    φ*, float
        J mol⁻¹ K⁻¹, reduced thermodynamical potential

    Notes
    -----
    Reduced thermdynamical potentials are calculated according to the equations

    .. math::
        \phi^* = Y_0 &+ Y \log(x) \\
                     &+ Y_{-2} x^{-2} \\
                     &+ Y_{-1} x^{-1} \\
                     &+ Y_1 x \\
                     &+ Y_2 x^2 \\
                     &+ Y_3 x^3

    where :math:`x \equiv 10^{-4} T`.

    References
    ----------
    Equation (82),
    Vargaftik, N B, and V S Yargin.
    Ch 7.4: Thermal Conductivity and Viscosity of the Gaseous Phase.
    Handbook of Thermodynamic and Transport Properties of Alkali Metals,
    edited by R. W. Ohse, 45. Blackwell Scientific Publications, 1985.

    $\phi^*$ coefficients are from the Table 35, page 823.
    """
    x = 1e-4 * T
    # functions = [1, log(x), 10^-4 x^-2, 10^-2 x^-1, x, x^2, x^3]
    table = [[187.7374, 19.5189, 4.7730, -6.117, 10.9728, -21.055, 21.357],
             [284.3545, 35.8511, -7.180, 17.819, 35.8331, -73.097, 46.625]]
    t = table[component - 1]
    phi = (t[0] + t[1] * log(x)
         + t[2] * 1e-4 * x**(-2)
         + t[3] * 1e-2 * x**(-1)
         + t[4] * x
         + t[5] * x**2
         + t[6] * x**3)
    return phi


def K_eq_Vargaftik_and_Yargin(T_gas, d_0_0=107800):
    """Equilibrium constant

    Parameters
    ----------
    T_gas : float
        K, gas temperature

    d_0_0 : float
        J/mol, dissociation energy at 0 K

    Notes
    -----
    :math:`\phi^*` from the coefficients in [1], Table 35, page 823.

    In the text, d_0_0 is written as :math:`D_0^0`

    Vargaftik and Voljak [2] give this constant as
    107800 ± 1300 J/mol or
    1.117 ± 0.012 eV/molecule

    References
    ----------
    .. [1] Vargaftik, N B, and V S Yargin.
       Ch 7.4: Thermal Conductivity and Viscosity of the Gaseous Phase.
       Handbook of Thermodynamic and Transport Properties of Alkali Metals,
       edited by R. W. Ohse, 45. Blackwell Scientific Publications, 1985.

    .. [2] Vargaftik, N. B.; Voljak, L. D.
       Ch 6.6.1: Thermodynamics Properties of Alkali Metal
       Vapours at Low Pressures. In: Handbook of Thermodynamic
       and Transport Properties of Alkali Metals;
       Ohse, R. W., Ed.; Blackwell Scientific Publications, 1985; p 12.
    """
    phi1 = phi_V_and_Y(1, T_gas)
    phi2 = phi_V_and_Y(2, T_gas)
    K_eq = exp((2 * phi1 - phi2) / R_gas - d_0_0 / (R_gas * T_gas))
    return K_eq


def x2_concentration_Vargaftik_and_Voljak(TK):
    """Diatomic molecule fraction in a saturated vapor

    Parameters
    ----------
    TK : float
        K, temperature

    Returns
    -------
    x₂ : float
        Dimer fraction

    References
    ----------
    Selected points from Table 2 of
    Vargaftik, N. B.; Voljak, L. D.
    Ch 6.6.1: Thermodynamics Properties of Alkali Metal
    Vapours at Low Pressures. In: Handbook of Thermodynamic
    and Transport Properties of Alkali Metals;
    Ohse, R. W., Ed.; Blackwell Scientific Publications, 1985; p 12.
    """
    data_T = [800, 850, 900, 1000, 1100, 1200, 1500, 1800, 2000]
    data_x2 = [
        0.007953, 0.01134, 0.0155, 0.02596, 0.03894, 0.05383, 0.1035, 0.1505,
        0.1767
    ]
    interp = interp1d(data_T, data_x2, kind='cubic', bounds_error=False)
    x_2 = interp(TK)
    return x_2


def eta1_Vargaftik_1991_Eq_6(TK):
    """Linear fit to monomer viscosity

    Parameters
    ----------
    TK : float
        K, temperature

    Returns
    -------
    η₁ : float
        Pa s, dynamic viscosity

    References
    ----------
    Equation (6) of
    Vargaftik, N. B., Yu. K. Vinogradov, V. I. Dolgov, V. G. Dzis,
    I. F. Stepanenko, Yu. K. Yakimovich, and V. S. Yargin.
    "Viscosity and Thermal Conductivity of Alkali Metal Vapors at
    Temperatures up to 2000 K." International Journal of
    Thermophysics 12, no. 1 (January 1991): 85–103.
    https://doi.org/10.1007/BF00506124.
    """
    return (129.1 + 0.100 * (TK - 1000)) * 1e-7


def eta_Vargaftik_1991_Eq_4(x2, TK):
    """Viscosity as a function of x2 and temperature

    Parameters
    ----------
    x2 : float
        Dimer fraction

    TK : float
        K, temperature

    Returns
    -------
    η : float
        Pa s, dynamic viscosity

    References
    ----------
    Vargaftik, N. B., Yu. K. Vinogradov, V. I. Dolgov, V. G. Dzis,
    I. F. Stepanenko, Yu. K. Yakimovich, and V. S. Yargin.
    "Viscosity and Thermal Conductivity of Alkali Metal Vapors at
    Temperatures up to 2000 K." International Journal of
    Thermophysics 12, no. 1 (January 1991): 85–103.
    https://doi.org/10.1007/BF00506124.
    """
    b1, b2, b3, b4 = 4.094, 3.335, 0.864, -6.964e-2
    numerator = 1 + b3 * x2 + b4 * x2**2
    denominator = 1 + b1 * x2 + b2 * x2**2
    eta = eta1_Vargaftik_1991_Eq_6(TK) * numerator / denominator
    return eta


def extrapolation_of_V_91_low_pressure(TK):
    Keq = K_eq_Vargaftik_and_Yargin(TK)
    P_kpa = press_best(TK) / 1000
    x2 = x2_concentration_Vargaftik_and_Yargin(P_kpa, Keq)
    eta_sat = eta_Vargaftik_1991_Eq_4(x2, TK)
    return eta_sat


def eta1_Vargaftik_1991_Table(TK):
    """Monomer viscosity

    Parameters
    ----------
    TK : float
        K, temperature

    Returns
    -------
    η₁ : float
        Pa s, viscosity of a monomer gas

    Notes
    ----
    Uncertainties: on the experiments:
    "Average error for the value thus obtained is estimated to be 5 %."

    References
    ----------
    Table IV of
    Vargaftik, N. B., Yu. K. Vinogradov, V. I. Dolgov, V. G. Dzis,
    I. F. Stepanenko, Yu. K. Yakimovich, and V. S. Yargin.
    "Viscosity and Thermal Conductivity of Alkali Metal Vapors at
    Temperatures up to 2000 K." International Journal of
    Thermophysics 12, no. 1 (January 1991): 85–103.
    https://doi.org/10.1007/BF00506124.
    """
    data_T = np.arange(800, 2600, 100)  # 800 to 2500
    data_eta1 = np.array([
        100, 112, 123, 134, 145, 155, 166, 176, 186, 196, 205, 215, 224, 233,
        242, 250, 260, 268
    ])
    interp = interp1d(data_T, data_eta1, kind='cubic', bounds_error=False)
    eta1 = 1e-7 * interp(TK)
    return eta1


def eta_sat_Vargaftik_1991_Table(TK):
    """Saturated vapor viscosity

    Parameters
    ----------
    TK : float
        K, temperature

    Returns
    -------
    η : float
        Pa s, viscosity of the saturated vapor

    Notes
    -----
    Uncertainties: on the experiments:
    "Average error for the value thus obtained is estimated to be 5 %."

    References
    ----------
    Table IV of
    Vargaftik, N. B., Yu. K. Vinogradov, V. I. Dolgov, V. G. Dzis,
    I. F. Stepanenko, Yu. K. Yakimovich, and V. S. Yargin.
    "Viscosity and Thermal Conductivity of Alkali Metal Vapors at
    Temperatures up to 2000 K." International Journal of
    Thermophysics 12, no. 1 (January 1991): 85–103.
    https://doi.org/10.1007/BF00506124.
    """
    data_T = np.arange(800, 2600, 100)  # 800 to 2500
    data_eta_sat = np.array([
        97.2, 106, 113, 118, 123, 126, 129, 131, 133, 135, 137, 139, 140, 141,
        143, 144, 146, 147
    ])
    interp = interp1d(data_T, data_eta_sat, kind='cubic', bounds_error=False)
    eta_sat = 1e-7 * interp(TK)
    return eta_sat


def eta1_Bouledroua(TK):
    """Viscosity of the monomers

    Parameters
    ----------
    TK : float
        K, temperature

    Returns
    -------
    η₁ : float
        Pa s, viscosity of a monomer gas

    References
    ----------
    Bouledroua, M, A Dalgarno, and R Côté.
    “Viscosity and Thermal Conductivity of Li, Na, and K Gases.”
    Physica Scripta 71, no. 5 (January 1, 2005): 519–22.
    https://doi.org/10.1238/Physica.Regular.071a00519.

    Notes
    -----
    "The numerical values in the Tables
    can be reproduced by the simple formula

    .. math::
        \eta = A T^{\alpha}, \quad A = 0.234, \quad \alpha = 0.903

    "
    """
    A = 0.234
    alpha = 0.903
    eta1 = 1e-7 * A * TK**alpha
    return eta1


def eta1_Bouledroua_Table_I(TK):
    """Viscosity of Li monomer gas

    Parameters
    ----------
    TK : float
        K, temperature

    Returns
    -------
    η₁ : float
        Pa s, viscosity of a monomer gas

    Notes
    -----
    Valid for 200 < TK < 2000.

    References
    ----------
    Bouledroua, M, A Dalgarno, and R Côté.
    “Viscosity and Thermal Conductivity of Li, Na, and K Gases.”
    Physica Scripta 71, no. 5 (January 1, 2005): 519–22.
    https://doi.org/10.1238/Physica.Regular.071a00519.

    Data from Tables I and IV. Eta there is in micropoise.
    """
    data_T = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    data_eta1 = [23, 49, 75, 100, 123, 144, 164, 184, 202, 221]

    interp = interp1d(data_T, data_eta1, kind='cubic', bounds_error=False)
    eta1 = 1e-7 * interp(TK)
    return eta1


def eta_Stepanenko(X2, T):
    """Dynamic viscosity

    Parameters
    ----------
    X2 : float
        Dimer fraction
    T : float
        K, temperature

    Returns
    -------
    η : float
        Pa s, viscosity of the gas

    References
    ----------
    Equation (5) of
    Stepanenko, I. F., N. I. Sidorov, Y. V. Tarlakov, and V. S. Yargin.
    “Experimental Study of the Viscosity of
        Lithium Vapor at High Temperatures.”
    International Journal of Thermophysics 7, no. 4 (July 1986): 829–35.
    https://doi.org/10.1007/BF00503840.

    Notes
    -----
    (the abstract suggests:) Valid for 1600 < T < 2000 K

    "Based on the analysis of the experimental errors, the accuracy of the
    data obtained has been estimated to be 3-4%."
    """
    eta = 1e-7 * (178 - 530 * (X2 - 0.05) + 0.071 * (T - 1700))
    eta_filtered = np.where(T >= 1500., eta, np.nan)
    return eta_filtered


def eta_Stepanenko_Table():
    """Viscosity of Li vapor

    Returns
    -------
    Array of temperature and viscosity data
    Units are K, Pa s

    Notes
    -----
    These tables contain points at various pressures,
    not necessarily at saturation

    References
    ----------
    Table I of
    Stepanenko, I. F., N. I. Sidorov, Y. V. Tarlakov, and V. S. Yargin.
    “Experimental Study of the Viscosity of
        Lithium Vapor at High Temperatures.”
    International Journal of Thermophysics 7, no. 4 (July 1986): 829–35.
    https://doi.org/10.1007/BF00503840.
    """
    data = np.array([[1595, 163],
                     [1607, 143],
                     [1668, 187],
                     [1692, 174],
                     [1700, 183],
                     [1715, 165],
                     [1722, 160],
                     [1747, 195],
                     [1812, 184],
                     [1815, 193],
                     [1823, 193],
                     [1852, 186],
                     [1970, 210],
                     [1983, 208]], dtype='float')
    data[:,1] = 1e-7 * data[:,1]
    return data

def eta1_Fialho_1993_Table(TK):
    """Monomer viscosity

    Parameters
    ----------
    TK : float
        K, temperature

    Returns
    -------
    η₁ : float
        Pa s, dynamic viscosity

    References
    ----------
    Fialho, P. S.; Fareleira, J. M. N. A.; Ramires, M. L. V.;
    Nieto de Castro, C. A.
    Thermophysical Properties of Alkali Metal Vapours, Part I.A.
    Ber. Bunsenges. Phys. Chem. 1993, 97 (11), 1487–1492.
    """
    data_T = np.arange(700,2100,100)
    data_eta1 = np.array([8.56,
                       9.71,
                       10.82,
                       11.89,
                       12.93,
                       13.93,
                       14.91,
                       15.86,
                       16.80,
                       17.72,
                       18.63,
                       19.53,
                       20.41,
                       21.30])
    interp = interp1d(data_T, data_eta1, kind='cubic', bounds_error=False)
    eta1 = 1e-6 * interp(TK)
    return eta1

# Thermal conductivity data
#---------------------------

def lambda1_Vargaftik_and_Yargin(TK):
    """Thermal conductivity of Li monomer gas

    Parameters
    ----------
    TK : float
        K, temperature

    Returns
    -------
    λ₁ : float
        W / m K, thermal conductivity

    References
    ----------
    Equation 66, page 822.
    Vargaftik, N B, and V S Yargin.
    Ch 7.4: Thermal Conductivity and Viscosity of the Gaseous Phase.
    Handbook of Thermodynamic and Transport Properties of Alkali Metals,
    edited by R. W. Ohse, 45. Blackwell Scientific Publications, 1985.

    Notes
    -----
    Valid 700 K < T < 2500 K

    Uncertainties: "for lithium vapour, ± 3 % for monatomic gas, and ± 7 % at the saturation line"
    """
    lambda1 = 1e-4 * (587.7 + 0.4562 * (TK - 1000) - 20.5e-6 * (TK - 1000)**2)
    return lambda1


def lambda_Vargaftik_and_Yargin(x2, TK):
    """Thermal conductivity with finite dimer fraction

    Parameters
    ----------
    x2 : float
        Mole fraction dimers in the gas
    TK : float
        K, temperature

    Returns
    -------
    λ : float
        W / m K, thermal conductivity

    References
    ----------
    Equations 65 and 67, page 822, of
    Vargaftik, N B, and V S Yargin.
    Ch 7.4: Thermal Conductivity and Viscosity of the Gaseous Phase.
    Handbook of Thermodynamic and Transport Properties of Alkali Metals,
    edited by R. W. Ohse, 45. Blackwell Scientific Publications, 1985.

    Notes
    -----
    Valid 700 K < T < 2500 K

    Uncertainties: "for lithium vapour, ± 3 % for monatomic gas, and ± 7 % at the saturation line"
    """
    T_r = 13583 + 0.297 * (TK - 1000) + 43e-6 * (TK - 1000)**2
    lambda1 = lambda1_Vargaftik_and_Yargin(TK)
    lambda_total = lambda1 * (1 - 3.84 * x2**1 + 13.6 * x2**2
                                  - 48 * x2**3 + 166 * x2**4
                                 - 576 * x2**5 + 1994 * x2**6
                                 + 0.095 * (T_r / TK)**2 * (x2 * (1 - x2))/((1 + x2)**2)
                             )
    return lambda_total

def lambda_sat_Vargaftik_and_Yargin_Table(TK):
    """Thermal conductivity of saturated vapor

    Parameters
    ----------
    TK : float
        K, temperature

    Returns
    -------
    λ : float
        W / m K, thermal conductivity

    References
    ----------
    Page 828, Table 37, of
    Vargaftik, N B, and V S Yargin.
    Ch 7.4: Thermal Conductivity and Viscosity of the Gaseous Phase.
    Handbook of Thermodynamic and Transport Properties of Alkali Metals,
    edited by R. W. Ohse, 45. Blackwell Scientific Publications, 1985.

    Notes
    -----
    Uncertainties: "for lithium vapour, [...] ± 7 % at the saturation line"
    """
    data_T = np.arange(700, 2025, 25)
    data_lambda_sats = np.array([
        497.2, 519.0, 541.6, 565.1, 589.2, 613.9, 638.8, 664.0, 689.2, 714.3,
        739.2, 763.6, 787.4, 810.6, 833.1, 854.6, 875.3, 895.0, 913.6, 931.2,
        947.8, 963.3, 977.7, 991.0, 1003.4, 1014.8, 1025.2, 1034.6, 1043.3,
        1051.0, 1058.0, 1064.3, 1069.9, 1074.8, 1079.1, 1082.9, 1086.1, 1088.9,
        1091.2, 1093.2, 1094.8, 1096.0, 1097.0, 1097.7, 1098.1, 1098.3, 1098.3,
        1098.2, 1097.8, 1097.3, 1096.7, 1096.0, 1095.1
    ])
    interp = interp1d(data_T,
                      data_lambda_sats,
                      kind='cubic',
                      bounds_error=False)
    lambda_sat = 1e-4 * interp(TK)
    return lambda_sat


def lambda1_Vargaftik_1991_Eq_5(TK):
    """Thermal conductivity of a monomer gas

    Parameters
    ----------
    TK : float
        K, temperature

    Returns
    -------
    λ₁ : float
        W / m K, thermal conductivity

    References
    ----------
    Equation (5) of
    Vargaftik, N. B., Yu. K. Vinogradov, V. I. Dolgov, V. G. Dzis,
    I. F. Stepanenko, Yu. K. Yakimovich, and V. S. Yargin.
    "Viscosity and Thermal Conductivity of Alkali Metal Vapors at
    Temperatures up to 2000 K." International Journal of
    Thermophysics 12, no. 1 (January 1991): 85–103.
    https://doi.org/10.1007/BF00506124.
    """
    lambda1 = (541.0 + 0.485 * (TK - 1000)) * 1e-4
    return lambda1


def lambda_Vargaftik_1991_Eq_3(x2, TK):
    """Thermal conductivity of a monomer-dimer mix

    Parameters
    ----------
    x2 : float
        Mole fraction dimers in the gas
    TK : float
        K, temperature

    Returns
    -------
    λ : float
        W / m K, thermal conductivity

    References
    ----------
    Vargaftik, N. B., Yu. K. Vinogradov, V. I. Dolgov, V. G. Dzis,
    I. F. Stepanenko, Yu. K. Yakimovich, and V. S. Yargin.
    "Viscosity and Thermal Conductivity of Alkali Metal Vapors at
    Temperatures up to 2000 K." International Journal of
    Thermophysics 12, no. 1 (January 1991): 85–103.
    https://doi.org/10.1007/BF00506124.
    """
    pass


def lambda1_Vargaftik_1991_Table(TK):
    """Thermal conductivity of a monomer gas

    Parameters
    ----------
    TK : float
        K, temperature

    Returns
    -------
    λ₁ : float
        W / m K, thermal conductivity

    References
    ----------
    Table III of
    Vargaftik, N. B., Yu. K. Vinogradov, V. I. Dolgov, V. G. Dzis,
    I. F. Stepanenko, Yu. K. Yakimovich, and V. S. Yargin.
    "Viscosity and Thermal Conductivity of Alkali Metal Vapors at
    Temperatures up to 2000 K." International Journal of
    Thermophysics 12, no. 1 (January 1991): 85–103.
    https://doi.org/10.1007/BF00506124.

    Notes
    -----
    Uncertainties: on the experiments:
    "Average error for the value thus obtained is estimated to be 5 %."
    """
    data_T = np.arange(800, 2600, 100)  # 800 to 2500
    data_lambda1 = np.array([
        450, 506, 558, 607, 655, 701, 745, 790, 834, 878, 921, 965, 1008, 1050,
        1092, 1131, 1169, 1203
    ])
    interp = interp1d(data_T, data_lambda1, kind='cubic', bounds_error=False)
    lambda1 = 1e-4 * interp(TK)
    return lambda1


def lambda_sat_Vargaftik_1991_Table(TK):
    """Thermal conductivity of saturated vapor

    Parameters
    ----------
    TK : float
        K, temperature

    Returns
    -------
    λ : float
        W / m K, thermal conductivity

    References
    ----------
    Table III of
    Vargaftik, N. B., Yu. K. Vinogradov, V. I. Dolgov, V. G. Dzis,
    I. F. Stepanenko, Yu. K. Yakimovich, and V. S. Yargin.
    "Viscosity and Thermal Conductivity of Alkali Metal Vapors at
    Temperatures up to 2000 K." International Journal of
    Thermophysics 12, no. 1 (January 1991): 85–103.
    https://doi.org/10.1007/BF00506124.
    """
    data_T = np.arange(800, 2600, 100)  # 800 to 2500
    data_lambda_sat = np.array([
        543, 652, 753, 841, 913, 966, 1003, 1029, 1045, 1055, 1058, 1058, 1054,
        1048, 1041, 1031, 1020, 1006
    ])
    interp = interp1d(data_T,
                      data_lambda_sat,
                      kind='cubic',
                      bounds_error=False)
    lambda_sat = 1e-4 * interp(TK)
    return lambda_sat


def lambda1_Bouledroua_Table(TK):
    """Thermal conductivity of the monomers

    Parameters
    ----------
    TK : float
        K, temperature

    Returns
    -------
    λ₁ : float
        W / m K, thermal conductivity

    Notes
    -----
    Valid for 200 < TK < 2000.

    References
    ----------
    Bouledroua, M, A Dalgarno, and R Côté.
    “Viscosity and Thermal Conductivity of Li, Na, and K Gases.”
    Physica Scripta 71, no. 5 (January 1, 2005): 519–22.
    https://doi.org/10.1238/Physica.Regular.071a00519.

    Data from Table V. Lambda in the table is 10^{-3} W / m K.
    """
    data = [[ 200, 10.31],
            [ 400, 21.97],
            [ 600, 33.63],
            [ 800, 44.84],
            [1000, 55.15],
            [1200, 64.57],
            [1400, 73.54],
            [1600, 82.51],
            [1800, 90.58],
            [2000, 99.10]]
    data = np.array(data)
    interp = interp1d(data[:, 0], data[:, 1], kind='cubic', bounds_error=False)
    lambda1 = 1e-3 * interp(TK)
    return lambda1

# Self-diffusion coefficients
#-----------------------------

def D11_Fialho_1993_Table(TK):
    """Self-diffusion coefficient at 0.10 MPa for monoatomic lithium.

    Parameters
    ----------
    TK : float
        K, temperature

    Returns
    -------
    D : float
        m² s⁻¹, self-diffusion coefficient

    References
    ----------
    Data from Table 3.
    Fialho, Paulo S., J.M.N.A. Fareleira, M.L.V. Ramires,
    and C.A. Nieto de Castro. "Thermophysical Properties
    of Alkali Metal Vapours, Part I.A."
    Berichte Der Bunsen-Gesellschaft Fur Physikalische Chemie
    97, no. 11 (1993): 1487–92.

    Notes
    -----
    Uncertainties: "The average collisions integrals
    :math:`\bar{\Omega}^{(l,s)}(T)` are compared with the results obtained
    previously [1] in Fig 2. The agreement is very good for lithium,
    where the maximum deviation is less than ±1%."
    """
    D_self_Fialho_1993_table = np.array([
        [700, 0.8885],
        [800, 1.1491],
        [900, 1.4393],
        [1000, 1.7589],
        [1100, 2.1077],
        [1200, 2.4859]])
    data = D_self_Fialho_1993_table
    interp = interp1d(data[:,0], data[:,1], kind='cubic', bounds_error=False)
    return 1e-4 * interp(TK)
