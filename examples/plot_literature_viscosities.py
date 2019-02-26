import numpy as np
import matplotlib.pyplot as plt
from lithdata.li_property_library import LiPropertyLibrary

lp = LiPropertyLibrary()

plt.style.use('seaborn-colorblind')

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(1,1,1)

colors = []
for i in range(5):
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

### Saturated, data
TK, eta = lp.eta_sat_Vargaftik_and_Yargin_Table().T
eta = yscale * eta
errors = (5/3) * lp.eta1_Vargaftik_and_Yargin_error(TK)
e_minus, e_plus = error_bands(eta, errors)
ax.plot(TK, eta, color=colors[0])
plt.fill_between(TK, e_minus, e_plus, alpha=0.2, color=colors[0])

## V&Y 1991
color = colors[3]
TK = np.linspace(800, 2000, 20)
eta1 = yscale * lp.eta1_Vargaftik_1991_Table(TK)
ax.plot(TK, eta1, label='Vargaftik 1991', color=color)
e_minus, e_plus = error_bands(eta1, 5)
plt.fill_between(TK, e_minus, e_plus, alpha=0.2, color=color)

### Saturated, data
eta_sat = yscale * lp.eta_sat_Vargaftik_1991_Table(TK)
e_minus, e_plus = error_bands(eta_sat, 5)
ax.plot(TK, eta_sat, color=color)
plt.fill_between(TK, e_minus, e_plus, alpha=0.2, color=color)

###Extrapolated saturated data using Equation 4.
#TK = np.linspace(600,2000,20)
#eta_sat = yscale * lp.extrapolation_of_V_91_low_pressure(TK)
#ax.plot(TK, eta_sat, color=color)

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
TK = np.linspace(200,2000,40)
eta1 = yscale * lp.eta1_Bouledroua_Table_I(TK)
ax.plot(TK, eta1, label = 'Bouledroua et al, 2005', color=colors[2])

## Fialho
TK = np.linspace(700,2000,40)
eta1 = yscale * lp.eta1_Fialho_1993_Table(TK)
ax.plot(TK, eta1, label = 'Fialho et al, 1993', color=colors[4])

ax.set_xlabel('T / K')
ax.set_ylabel('$\eta$ / (µPa s)')
plt.annotate('Monomers', xy=(0.68, 0.65), xycoords='axes fraction')
plt.annotate('Saturated', xy=(0.65, 0.48), xycoords='axes fraction')
plt.title('Viscosity of lithium vapor')

ax.legend(loc=4)
plt.show()
