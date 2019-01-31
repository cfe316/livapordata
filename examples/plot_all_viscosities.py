import numpy as np
import matplotlib.pyplot as plt
from lithdata.li_property_library import LiPropertyLibrary

lp = LiPropertyLibrary()

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(1,1,1)

colors = []
for i in range(3):
    color=next(ax._get_lines.prop_cycler)['color']
    colors.append(color)

def error_bands(value, percent):
    e_plus = (1 + 0.01 * percent) * value
    e_minus = (1 - 0.01 * percent) * value
    return e_minus, e_plus

yscale = 1e6

## V & Y
### monomers
TK = np.linspace(700, 2500, 20)
eta1 = yscale * lp.eta1_Vargaftik_and_Yargin(TK)
ax.plot(TK, eta1, label='Vargaftik and Yargin 1985', color=colors[0])
errors = lp.eta1_Vargaftik_and_Yargin_error(TK)
e_minus, e_plus = error_bands(eta1, errors)
plt.fill_between(TK, e_minus, e_plus, alpha=0.2, color=colors[0])

### Saturated
TK = np.linspace(1057,2156, 20)
pressures_kpa = lp.vapor_pressure_Browning_and_Potter(TK)/1000.
keqs = lp.K_eq_Vargaftik_and_Yargin(TK)
x2 = lp.x2_concentration_Vargaftik_and_Yargin(pressures_kpa, keqs)
eta = yscale * lp.eta_Vargaftik_and_Yargin(x2, TK)
ax.plot(TK, eta, color=colors[0])
# Errors about 5% at the saturation line vs 3% for monomers.
errors = (5/3) * lp.eta1_Vargaftik_and_Yargin_error(TK)
e_minus, e_plus = error_bands(eta, errors)
plt.fill_between(TK, e_minus, e_plus, alpha=0.2, color=colors[0])


## Stepanenko
### monomers
TK = np.linspace(1500,2000,20)
eta1 = yscale * lp.eta_Stepanenko(0,TK)
ax.plot(TK, eta1, label = 'Stepanenko et al, 1986', color=colors[1])
e_minus, e_plus = error_bands(eta1, 3.5)
plt.fill_between(TK, e_minus, e_plus, alpha=0.2, color=colors[1])

### Saturated
TK = np.linspace(1057,2000, 20)
# This seems to be the best source of vapor pressure data
pressures_kpa = lp.vapor_pressure_Browning_and_Potter(TK)/1000.
# This seems to be the best source data on Keq and the x2 fraction
keqs = lp.K_eq_Vargaftik_and_Yargin(TK)
x2 = lp.x2_concentration_Vargaftik_and_Yargin(pressures_kpa, keqs)

eta = yscale * lp.eta_Stepanenko(x2,TK)
ax.plot(TK, eta, color=colors[1])
e_minus, e_plus = error_bands(eta, 3.5)
plt.fill_between(TK, e_minus, e_plus, alpha=0.2, color=colors[1])

## Bouledroua
TK = np.linspace(200,2000,20)
eta1 = yscale * lp.eta1_Bouledroua_Table_I(TK)
ax.plot(TK, eta1, label = 'Bouledroua et al, 2005', color=colors[2])

ax.set_xlabel('T / K')
ax.set_ylabel('$\eta$ / (µPa s)')

ax.legend(loc=4)
plt.show()
