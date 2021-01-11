import numpy as np
from livapordata.constants import kB
from livapordata.basics import mLi
from livapordata.utility import langmuir_flux
from livapordata.vaporpressure import press_best

class LithiumProperties():

    def __init__(self):
        self.m = mLi
        heatVap = 147.0 / 6.022e23 # heat of vaporization in kJ per lithium atom (source?)
        kj_to_j = 1e3 # kilojoules to joules
        self.heat_vap = heatVap * kj_to_j

    def vapor_pressure(self, t_kelvin):
        """Vapor pressure at a given temperature.

        Parameters
        ----------
        t_kelvin, K

        Returns
        -------
        vapor pressure, Pa

        Notes
        -----
        Uses the recommended 'best' vapor pressure curve.
        This currently the one by

        Browning, P., and Potter, P. E. “Assessment of the Experimentally
        Determined Vapour Pressures of the Liquid Alkali Metals.”
        In Handbook of Thermodynamic and Transport Properties of Alkali Metals,
        349–58. Oxford: Blackwell Scientific Publications, 1985.

        See vaporpressure.py for additional information.
        """
        return press_best(t_kelvin)

    def vapor_number_density(self, t_kelvin):
        """Equilibrium lithium vapor number density.

        Assumes ideal gas.

        Parameters
        ----------
        t_kelvin, K

        Returns
        -------
        Vapor number density, m⁻³
        """
        t_kelvin = 1.0 * np.array(t_kelvin)
        p = t_kelvin > 0
        res = np.zeros_like(t_kelvin)
        res[p] = self.vapor_pressure(t_kelvin[p]) / (kB * t_kelvin[p])
        density = res
        return density

    def vapor_mass_density(self, t_kelvin):
        """Equilibrium lithium vapor density in kg/m^3.

        Parameters
        ----------
        t_kelvin, K

        Returns
        -------
        Density, kg/m³
        """
        t_kelvin = 1.0 * np.array(t_kelvin)
        p = t_kelvin > 0
        res = np.zeros_like(t_kelvin)
        res[p] = self.vapor_pressure(t_kelvin[p]) / (kB * t_kelvin[p])
        density = res
        return density

    def langmuir_flux(self, temperature):
        """Calculates the equilibrium Langmuir flux of stationary vapor.

        Parameters
        ----------
        temperature, K

        Returns
        -------
        Vapor flux, m⁻² s⁻¹
        """
        temperature = 1.0 * np.array(temperature)
        density = self.vapor_number_density(temperature)
        flux = langmuir_flux(density, temperature, self.m)
        return flux
