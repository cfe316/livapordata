import numpy as np
from numpy import sqrt, log, log10, exp, pi
from lithdata.constants import TORR_TO_PASCALS, BARS_TO_PASCALS, ATM_TO_PASCALS

def press_best(t_k):
    """Best estimate for vapor pressure"""
    return press_Browning_and_Potter(t_k)

def press_Browning_and_Potter(t_k):
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
    p_megapascals = exp(c1 + c2 / t_k+ c3 * log(t_k))
    pressure_pa = p_megapascals * megapascals_to_pascals
    return pressure_pa

def press_NIST_webbook(t_k):
    """
    https://webbook.nist.gov/cgi/cbook.cgi?ID=C7439932&Mask=4&Type=ANTOINE&Plot=on
    A fit to the Antoine equation.
    "Coefficients calculated by NIST from author's data:

    Hicks, W.T. "Evaluation of Vapor-Pressure Data for Mercury,Lithium,Sodium,and Potassium."
    The Journal of Chemical Physics 38,no.8 (April 15,1963):1873-1880.
    https://doi.org/10.1063/1.1733889. 
    "

    Applicable range: 298.14 K to 1599.99 K
    """
    a = 4.98831
    b = 7918.984
    c = -9.52
    p_bars = 10 ** (a - b / (t_k  + c))
    p_pa = BARS_TO_PASCALS * p_bars
    return p_pa

def press_Davison_1968(t_k):
    """
    Davison, H.W. 
    "Complication of Thermophysical Properties of Liquid Lithium."
    NASA Technical Note TN D-4650. Cleveland, Ohio: Lewis Research Center, July 1968.

    "With this equation the data shown in figure 8 are correlated
     with a standard deviation of 3.38 percent. The maximum deviation
     of -32.6 percent occurred at a vapor pressure of about 6 Pa."

    The plot Figure 8 shows 800 to 1800 K so we can take that as the applicable range.
    """
    return 10 ** (10.015 - 8064.5 / t_k)

def press_Maucherat_1939(t_k):
    """
    M. Maucherat, Pression de vapeur saturante du lithium entre 462°c et 642°c.
    Journal de Physique et le Radium, Volume 10, Issue 10, pages 441 - 444.
    
    Applicable range: 462°C to 642°C = 735 K to 915 K.
    """
    p_torr = 10 ** (8.012 - 8172 / t_k)
    p_pa = TORR_TO_PASCALS * p_torr
    return p_pa

def press_Vargaftik_and_Kapitonov(t_k):
    """
    Vargaftik, N. B., V. M. Kapitonov, and A. A. Voshchinin
    "Experimental Study of the Thermal Conductivity of Lithium Vapor."
    Journal of Engineering Physics 49, no. 4 (October 1985): 1208-1212.
    https://doi.org/10.1007/BF00871920.

    For the pressure curve this paper cites:

    V. S. Yargin, N. I. Sidorov, E. L. Studnikov, and Yu. K. Vinogradov,
    "Transport properties of saturated lithium vapor,"
    Inzh.-Fiz. Zh., 43, No. 3, 494 (1982).

    I don't have that paper. Experiments by Vargaftik were performed over
    1226 K to 1415 K, so we could take that as the applicable range.
    """
    p_atm = 10 ** (+8.5088 - 8363 / t_k - 1.02573 * log10(t_k) 
                   -1.3091e-4 * t_k + 1.08872 * exp(-2940 / t_k))
    p_pa = ATM_TO_PASCALS * p_atm

def press_Golubchikov(t_k):
    """
    Golubchikov, L. G., V. A. Evtikhin, I. E. Lyublinski, V. I. Pistunovich, 
    I. N. Potapov, and A. N. Chumanov. 
    "Development of a Liquid-Metal Fusion Reactor Divertor 
        with a Capillary-Pore System." 
    Journal of Nuclear Materials 233 (1996): 667-672.

    Reported for a temperature range of: 180-1300°C = 453 K to 1573 K.

    Note that this formula is cited as coming from one of two Russian papers:

    Bystrov, P.I., D. N. Kagan, G. A. Krechtova and E. E. Shpilrain, 
    Liquid Metal Heat Carriers, Thermal Pipes and Power Installations
    Science, Moscow, 1988

    -and/or-

    Griaznov, G.M. et al, Materials in Liquid Metal Systems of Fusion Reactors.
    Energoatomizdat, Moscow, 1989.
    """
    p_pa = 10 ** (12.4037 - 8283.1 / t_k - 0.7081 * log10(t_k))
    return p_pa
