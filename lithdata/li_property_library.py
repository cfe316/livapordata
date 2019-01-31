import numpy as np
from numpy import sqrt, log, exp
from scipy.interpolate import interp1d

class LiPropertyLibrary():

    def __init__(self):
        pass

    ### Vapor pressure
    def vapor_pressure_Browning_and_Potter(self, t_kelvin):
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

    ### Transport properties: viscosity, thermal conductivity, diffusivity

    def eta1_Vargaftik_and_Yargin(self, TK):
        """
        Viscosity of the monomers as a function of temperature.
        
        Vargaftik, N B, and V S Yargin. 
        Ch 7.4: Thermal Conductivity and Viscosity of the Gaseous Phase.
        Handbook of Thermodynamic and Transport Properties of Alkali Metals,
        edited by R. W. Ohse, 45. Blackwell Scientific Publications, 1985.

        Equation (56), page 821."""
        eta1 = 1e-7 * (130.6 + 0.1014 * (TK - 1000) - 4.55e-6 * (TK - 1000)**2)
        return eta1

    def eta1_Vargaftik_and_Yargin_error(self, TK):
        """
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
        return y1 + (y2 - y1) * (TK - x1)/(x2 - x1)

    def eta_Vargaftik_and_Yargin(self, x2, TK):
        """Viscosity with two components, monomer and dimer.
            Equation (55), page 821.
            $$ \eta(x_2, T) = \eta_1(T) \left(1 - 3.65 x_2 
                                     + 12.5 x_2^2 - 42 x_2^3 
                                     + 142 x_2^4 - 479 x_2^5 
                                     + 1600 x_2^6\right)$$
        """
        eta1 = self.eta1_Vargaftik_and_Yargin(TK)
        eta = eta1 * (1 - 3.65 * x2    + 12.5 * x2**2
                          - 42 * x2**3 +  142 * x2**4 
                         - 479 * x2**5 + 1600 * x2**6)
        return eta

    def x2_concentration_Vargaftik_and_Yargin(self, P_kpa, Keq):
        """Equation (80)
        """
        x2 = 1 - 2/(1 + sqrt(1 + 3.9477e-2 * P_kpa / Keq))
        return x2

    def phi_V_and_Y(self, component, T):
        """
        Vargaftik, N B, and V S Yargin. 
        Ch 7.4: Thermal Conductivity and Viscosity of the Gaseous Phase.
        Handbook of Thermodynamic and Transport Properties of Alkali Metals,
        edited by R. W. Ohse, 45. Blackwell Scientific Publications, 1985.

        $\phi^*$ from the coefficients in Table 35, page 823.
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

    def K_eq_Vargaftik_and_Yargin(self, T_gas, d_0_0=107800):
        """
        Vargaftik, N B, and V S Yargin. 
        Ch 7.4: Thermal Conductivity and Viscosity of the Gaseous Phase.
        Handbook of Thermodynamic and Transport Properties of Alkali Metals,
        edited by R. W. Ohse, 45. Blackwell Scientific Publications, 1985.

        $\phi^*$ from the coefficients in Table 35, page 823.
        """
        R_gas = 8.31446
        phi1 = self.phi_V_and_Y(1, T_gas)
        phi2 = self.phi_V_and_Y(2, T_gas)
        K_eq = exp((2 * phi1 - phi2)/R_gas - d_0_0 / (R_gas * T_gas))
        return K_eq

    # Vargaftik and Voljak table
    def x2_concentration_Vargaftik_and_Voljak(self, TK):
        data_T = [800, 850, 900, 1200, 1500, 1800, 2000]
        data_x2 = [0.007953, 0.01134, 0.0155, 0.05383, 0.1035, 0.1505, 0.1767]
        interp = interp1d(data_T, data_x2, kind='cubic', bounds_error=False)
        x_2 = interp(TK)
        return x_2

    # Bouledroua et al, 2005 Phys. Scr. 71 519
    def eta1_Boulederoua(self, TK):
        """
        Viscosity of the monomers as a function of temperature.

        Bouledroua, M, A Dalgarno, and R Côté.
        “Viscosity and Thermal Conductivity of Li, Na, and K Gases.”
        Physica Scripta 71, no. 5 (January 1, 2005): 519–22. 
        https://doi.org/10.1238/Physica.Regular.071a00519.
        
        "The numerical values in the Tables
        can be reproduced by the simple formula
        $$ \eta = A T^{\alpha}, \quad A = 0.234, \quad \alpha = 0.903$$"

        """
        A = 0.234
        alpha = 0.903
        eta1 = 1e-7 * A * TK**alpha
        return eta1

    def eta1_Bouledroua_Table_I(self, TK):
        """
        Parameters:
            TK, temperature in Kelvin. 200 < TK < 2000.

        Viscosity of the monomers as a function of temperature.

        Bouledroua, M, A Dalgarno, and R Côté.
        “Viscosity and Thermal Conductivity of Li, Na, and K Gases.”
        Physica Scripta 71, no. 5 (January 1, 2005): 519–22. 
        https://doi.org/10.1238/Physica.Regular.071a00519.

        Data from Tables I and IV. Eta in micropoise.
        """
        data_T = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
        data_eta1 = [23, 49, 75, 100, 123, 144, 164, 184, 202, 221]
        
        interp = interp1d(data_T, data_eta1, kind='linear', bounds_error=False)
        eta1 = 1e-7 * interp(TK)
        return eta1

    # Stepanenko et al, Experimental $\eta$ at high temperatures, 1986
    def eta_Stepanenko(self, X2, T):
        """
        Stepanenko, I. F., N. I. Sidorov, Y. V. Tarlakov, and V. S. Yargin.
        “Experimental Study of the Viscosity of 
            Lithium Vapor at High Temperatures.” 
        International Journal of Thermophysics 7, no. 4 (July 1986): 829–35.
        https://doi.org/10.1007/BF00503840.

        Equation (5).

        (the abstract suggests:) Valid for 1600 < T < 2000 K

        "Based on the analysis of the experimental errors, the accuracy of the
        data obtained has been estimated to be 3-4%."
        """
        eta = 1e-7 * (178 - 530 * (X2 - 0.05) +0.071 * (T - 1700))
        eta_filtered = np.where(T >= 1500., eta, np.nan)
        return eta_filtered
    
    def eta_Stepanenko_Table(self):
        """
        These tables are points at various pressures, probably not saturated.

        Stepanenko, I. F., N. I. Sidorov, Y. V. Tarlakov, and V. S. Yargin.
        “Experimental Study of the Viscosity of 
            Lithium Vapor at High Temperatures.” 
        International Journal of Thermophysics 7, no. 4 (July 1986): 829–35.
        https://doi.org/10.1007/BF00503840.

        Data from Table I.
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

    # VHS model from Bird:
    def eta_Bird_VHS(self, T, vhs_model):
        """Bird 2013, Chapter 2, Eq 43
        """
        m, dref, omega, Tref = vhs_model
        mu_ref_numerator = (15 * sqrt(kB * m * Tref / pi))
        mu_ref_denominator = 2 * dref **2 * (7 - 2 * omega) * (5 - 2 * omega)
        mu_ref = mu_ref_numerator / mu_ref_denominator
        
        mu = mu_ref * (T / Tref) ** omega
        
        return mu

    def eta_Bird_VSS(self, T, vss_model):
        """
        Bird 2013, Chapter 3, Eq 19, solved for mu.
        """
        m, dref, omega, Tref, alpha = vss_model
        
        mu_ref_numerator = (5 * (1 + alpha) * (2 + alpha) * sqrt(kB * m * Tref / pi))
        mu_ref_denominator = 4 * dref **2 * (7 - 2 * omega) * (5 - 2 * omega)
        mu_ref = mu_ref_numerator / mu_ref_denominator
        
        mu = mu_ref * (T / Tref) ** omega
        return mu

    #### Thermal conductivity data
    def lambda1_Vargaftik_and_Yargin(self, TK):
        """
        Vargaftik, N B, and V S Yargin. 
        Ch 7.4: Thermal Conductivity and Viscosity of the Gaseous Phase.
        Handbook of Thermodynamic and Transport Properties of Alkali Metals,
        edited by R. W. Ohse, 45. Blackwell Scientific Publications, 1985.

        Equation 66, page 822.
        Valid 700 K < T < 2500 K
        
        Uncertainties: "for lithium vapour, ± 3 % for monatomic gas, and ± 7 % at the saturation line"
        """
        lambda1 = 1e-4 * (587.7 + 0.4562 * (TK - 1000) - 20.5e-6 * (TK - 1000)**2)
        return lambda1

    def lambda_Vargaftik_and_Yargin(self, x2, TK):
        """
        Vargaftik, N B, and V S Yargin. 
        Ch 7.4: Thermal Conductivity and Viscosity of the Gaseous Phase.
        Handbook of Thermodynamic and Transport Properties of Alkali Metals,
        edited by R. W. Ohse, 45. Blackwell Scientific Publications, 1985.

        Equations 65 and 67, page 822.
        Valid 700 K < T < 2500 K

        Uncertainties: "for lithium vapour, ± 3 % for monatomic gas, and ± 7 % at the saturation line"
        """
        T_r = 13583 + 0.297 * (TK - 1000) + 43e-6 * (TK - 1000)**2
        lambda1 = self.lambda1_Vargaftik_and_Yargin(TK)
        lambda_total = lambda1 * (1 - 3.84 * x2**1 + 13.6 * x2**2
                                      - 48 * x2**3 + 166 * x2**4
                                     - 576 * x2**5 + 1994 * x2**6 
                                     + 0.095 * (T_r / TK)**2 * (x2 * (1 - x2))/((1 + x2)**2)
                                 )
        return lambda_total

    def lambda1_Bouledroua_Table(self, TK):
        """
        Parameters:
            TK, temperature in Kelvin. 200 < TK < 2000.

        Coefficients of thermal conductivity of the monomers.

        Bouledroua, M, A Dalgarno, and R Côté.
        “Viscosity and Thermal Conductivity of Li, Na, and K Gases.”
        Physica Scripta 71, no. 5 (January 1, 2005): 519–22. 
        https://doi.org/10.1238/Physica.Regular.071a00519.

        Data from Table V. Lambda in the table is 10^{-3} W/mK.
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
        interp = interp1d(data[:,0], data[:,1], kind='linear', bounds_error=False)
        lambda1 = 1e-3 * interp(TK)
        return lambda1

    def K_Bird_VHS(self, T, vhs_model):
        """Bird 2013, Chapter 2, Equation 44.
        """
        m, dref, omega, Tref = vhs_model
        mu = eta_Bird_VHS(T, vhs_model)
        K = (15/4) * kB * mu / m
        return K

    #### Self-diffusion coefficients
    def D11_Fialho_1993_Table(self, TK):
        """
        Self-diffusion coefficient at 0.10 MPa for monoatomic lithium.

        Fialho, Paulo S., J.M.N.A. Fareleira, M.L.V. Ramires,
        and C.A. Nieto de Castro. "Thermophysical Properties
        of Alkali Metal Vapours, Part I.A." 
        Berichte Der Bunsen-Gesellschaft Fur Physikalische Chemie
        97, no. 11 (1993): 1487–92.

        Uncertainties: "The average collisions integrals 
        $\bar{\Omega}^{(l,s)}(T)$ are compared with the results obtained 
        previously [1] in Fig 2. The agreement is very good for lithium,
        where the maximum deviation is less than ±1%."

        Data from Table 3.
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


