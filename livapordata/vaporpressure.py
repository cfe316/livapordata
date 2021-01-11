import numpy as np
from numpy import sqrt, log, log10, exp, pi
from livapordata.constants import TORR_TO_PASCALS, BARS_TO_PASCALS, ATM_TO_PASCALS

def press_best(t_k):
    """Best estimate for vapor pressure"""
    return press_Browning_and_Potter(t_k)

def press_Alcock(t_k):
    """From Alcock et al.

    Parameters
    ----------
    t_k, K

    Returns
    -------
    Vapor pressure, Pa

    References
    ----------
    Alcock, C. B., Itkin, V. P., Horrigan, M. K.  "Vapour pressure equations for
    the metallic elements 298-2500K."
    Canadian metallurgical quarterly, Vol 23, No 3, Pages 309-313, 1984.

    I'm fairly certain this is one of the 'practical' equations,
    for which they claim +- 5% accuracy.

    For this formula, Alcock cites an unpublished paper by
    themself and V. P. Itkin, as well as

    L. V. Gurvich, I. V. Veits and V. A. Medvedev et al.,
    Termodinamicheskie Svoistva Indidual'nykh Veshchestv
    (Thermodynamic properties of individual substances), Vols 1-4.
    Izdatel'stvo "Nauka", Moscow (1978 - 1982)

    Notes
    -----
    Equation for Li liquid.
    Valid from the melting point to 1000K.

    """
    A = 5.055
    B = -8023
    p_atm = 10**(A + B / t_k)
    p_pa = ATM_TO_PASCALS * p_atm
    return p_pa


def press_Bohdansky(t_k):
    """From Bohdansky

    Parameters
    ----------
    t_k, K

    Returns
    -------
    Vapor pressure, Pa

    References
    ----------
    Bohdansky, J., Vapor pressure of different metals in
    the pressure range of 50 to 4000 torr.
    The Journal of Physical Chemistry, Volume 71, Issue 2,
    pages 215-217. Jan 1967.

    Coefficients in Table III.

    Notes
    -----
    "Data ranged from 1374 K to 1881 K."

    """
    p_torr = 10**(7.67 - 7740 / t_k)
    p_pa = TORR_TO_PASCALS * p_torr
    return p_pa


def press_Browning_and_Potter(t_k):
    """From Browning and Potter

    Parameters
    ----------
    t_k, K

    Returns
    -------
    Vapor pressure, Pa


    References
    ----------
    Browning, P., and Potter, P. E. “Assessment of the Experimentally
    Determined Vapour Pressures of the Liquid Alkali Metals.”
    In Handbook of Thermodynamic and Transport Properties of Alkali Metals,
    349–58. Oxford: Blackwell Scientific Publications, 1985.

    Section 6.2, Page 350, Equation (2)

    Notes
    -----
    Valid over 1057 K < T < 2156 K.

    An expression for uncertainty is given but it's not entirely
    clear what it means.
    The equation is written as

    ln P(MPa) = 13.0719 ± 1.8424 - (18880.659 ± 347.220)/T(K)
                - 0.4942 ± 0.2208 ln T(K)
    """
    c1 = 13.0719
    c2 = -18880.659
    c3 = -0.4942
    megapascals_to_pascals = 1e6
    p_megapascals = exp(c1 + c2 / t_k + c3 * log(t_k))
    pressure_pa = p_megapascals * megapascals_to_pascals
    return pressure_pa


def press_NIST_webbook(t_k):
    """From the NIST Webbook

    Parameters
    ----------
    t_k, K

    Returns
    -------
    Vapor pressure, Pa

    References
    ----------
    A fit to the Antoine equation.
    https://webbook.nist.gov/cgi/cbook.cgi?ID=C7439932&Mask=4&Type=ANTOINE&Plot=on
    "Coefficients calculated by NIST from author's data:

    Hicks, W.T. "Evaluation of Vapor-Pressure Data for Mercury,Lithium,Sodium,and Potassium."
    The Journal of Chemical Physics 38,no.8 (April 15,1963):1873-1880.
    https://doi.org/10.1063/1.1733889.
    "

    Notes
    -----
    Applicable range: 298.14 K to 1599.99 K
    """
    a = 4.98831
    b = 7918.984
    c = -9.52
    p_bars = 10**(a - b / (t_k + c))
    p_pa = BARS_TO_PASCALS * p_bars
    return p_pa


def press_Davison_1968(t_k):
    """From Davison

    Parameters
    ----------
    t_k, K

    Returns
    -------
    Vapor pressure, Pa

    References
    ----------
    Davison, H.W.
    "Complication of Thermophysical Properties of Liquid Lithium."
    NASA Technical Note TN D-4650. Cleveland, Ohio: Lewis Research Center, July 1968.

    Notes
    -----
    "With this equation the data shown in figure 8 are correlated
     with a standard deviation of 3.38 percent. The maximum deviation
     of -32.6 percent occurred at a vapor pressure of about 6 Pa."

    The plot Figure 8 shows 800 to 1800 K so we can take that as the applicable range.
    """
    return 10**(10.015 - 8064.5 / t_k)


def press_Maucherat_1939(t_k):
    """From Maucherat

    Parameters
    ----------
    t_k, K

    Returns
    -------
    Vapor pressure, Pa

    References
    ----------
    M. Maucherat, Pression de vapeur saturante du lithium entre 462°c et 642°c.
    Journal de Physique et le Radium, Volume 10, Issue 10, pages 441 - 444.

    Notes
    -----
    Applicable range: 462°C to 642°C = 735 K to 915 K.
    """
    p_torr = 10**(8.012 - 8172 / t_k)
    p_pa = TORR_TO_PASCALS * p_torr
    return p_pa

def press_Yargin_and_Sidorov(t_k):
    """From Yargin and Sidorov

    Parameters
    ----------
    t_k, K

    Returns
    -------
    Vapor pressure, Pa

    References
    ----------
    V. S. Yargin, N. I. Sidorov, E. L. Studnikov, and Yu. K. Vinogradov,
    "Transport properties of saturated lithium vapor,"
    Inzh.-Fiz. Zh., 43, No. 3, 494 (1982).

    Notes
    -----
    This paper is in Russian. The equation was synthesized
    from experiments at pressures between 253 Pa and 1.25e5 Pa,
    roughly 1050 K to 1700 K.

    No estimate of the uncertainty is provided.
    """
    p_atm = 10**(+8.5088 - 8363 / t_k - 1.02573 * log10(t_k) -
                 1.3091e-4 * t_k + 1.08872 * exp(-2940 / t_k))
    p_pa = ATM_TO_PASCALS * p_atm
    return p_pa

def press_Golubchikov(t_k):
    """As reported in Golubchikov

    Parameters
    ----------
    t_k, K

    Returns
    -------
    Vapor pressure, Pa

    References
    ----------
    Golubchikov, L. G., V. A. Evtikhin, I. E. Lyublinski, V. I. Pistunovich,
    I. N. Potapov, and A. N. Chumanov.
    "Development of a Liquid-Metal Fusion Reactor Divertor
        with a Capillary-Pore System."
    Journal of Nuclear Materials 233 (1996): 667-672.

    Reported for a temperature range of: 180-1300°C = 453 K to 1573 K.

    Note that this formula is cited as coming from one of two Russian books / papers:

    Bystrov, P.I., D. N. Kagan, G. A. Krechtova and E. E. Shpilrain,
    Liquid Metal Heat Carriers, Thermal Pipes and Power Installations
    Science, Moscow, 1988

    -and/or-

    Griaznov, G.M. et al, Materials in Liquid Metal Systems of Fusion Reactors.
    Energoatomizdat, Moscow, 1989.
    (I suspect the second.)
    """
    p_pa = 10**(12.4037 - 8283.1 / t_k - 0.7081 * log10(t_k))
    return p_pa

def press_Bystrov(t_k):
    """From Bystrov

    Parameters
    ----------
    t_k, K

    Returns
    -------
    Vapor pressure, Pa

    References
    ----------
    Bystrov, P.I., D. N. Kagan, G. A. Krechtova and E. E. Shpilrain,
    Liquid Metal Heat Carriers, Thermal Pipes and Power Installations
    Science, Moscow, 1988

    Notes
    -----
    Applicable range: 700 K to 2000 K.
    "The error is \delta P = 2%"
    """
    tau = t_k * 1.0e-3
    c = -2.0532
    am1 = -19.4268
    a0 = 9.4993
    a1 = 0.7530

    p_megapascals = exp(c * log(tau) + am1 / tau + a0 + a1 * tau)
    megapascals_to_pascals = 1e6
    pressure_pa = p_megapascals * megapascals_to_pascals
    return pressure_pa

def press_JSME_data_book(t_k):
    """From the JSME data book

    Parameters
    ----------
    t_k, K

    Returns
    -------
    Vapor pressure, Pa

    References
    ----------
    Antoine equation fit, found in

    The Japan Society of Mechanical Engineers (JSME),
    JSME Data Book: Heat Transfer, fifth ed.,
    Maruzen, Tokyo, 2009 (in Japanese)

    (I don't directly have this book.)

    Notes
    -----
    Cited by:
    Kanemura et al, Analytical and experimental study of
    the evaporation and deposition rates from a high-speed
    liquid lithium jet. Fusion Engineering and Design,
    122, pages 176-185, November, 2017.
    """
    a = 9.94079
    b = -8001.8
    c = 6.676
    pressure_pa = 10**(a + b / (t_k + c))
    return pressure_pa
