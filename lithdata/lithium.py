import numpy as np
from lithdata.constants import kB
from lithdata.basics import mLi
from lithdata.utility import langmuir_flux
from lithdata.vaporpressure import press_best

class LithiumProperties():

    def __init__(self):
        #self.m = 1.1526e-26 # this exactly matches the mass in SPARTA, air.vss. 
        self.m = mLi # this exactly matches the mass in SPARTA, air.vss. 
        heatVap = 147.0 / 6.022e23 # heat of vaporization in kJ per lithium atom (source?)
        kj_to_j = 1e3 # kilojoules to joules
        self.heat_vap = heatVap * kj_to_j

    def vapor_pressure(self, t_kelvin):
        return press_best(t_kelvin)

    def vapor_number_density(self, t_kelvin):
        """Equilibrium lithium vapor density in #/m^3.

        Assumes ideal gas.

        Parameters: T in Kelvins
        """
        t_kelvin = 1.0 * np.array(t_kelvin)
        p = t_kelvin > 0
        res = np.zeros_like(t_kelvin)
        res[p] = self.vapor_pressure(t_kelvin[p]) / (kB * t_kelvin[p])
        density = res
        return density
 
    def vapor_mass_density(self, t_kelvin):
        """Equilibrium lithium vapor density in kg/m^3.

        Parameters: T in Kelvins
        """
        t_kelvin = 1.0 * np.array(t_kelvin)
        p = t_kelvin > 0
        res = np.zeros_like(t_kelvin)
        res[p] = self.vapor_pressure(t_kelvin[p]) / (kB * t_kelvin[p])
        density = res
        return density

    def langmuir_flux(self, temperature):
        temperature = 1.0 * np.array(temperature)
        """Calculates the equilibrium Langmuir flux in # m^{-2} s^{-1}."""
        density = self.vapor_number_density(temperature)
        flux = langmuir_flux(density, temperature, self.m)
        return flux
