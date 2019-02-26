import numpy as np
import matplotlib.pyplot as plt
from lithdata.li_property_library import LiPropertyLibrary
from lithdata.fit_dsmc_vhs import VHS_model_from_viscosity
from lithdata.constants import mLi

lp = LiPropertyLibrary()

visc_func_to_fit = lp.eta1_Bouledroua_Table_I
mass = mLi
T_min, T_max = 700, 1000
vhs_b = VHS_model_from_viscosity(visc_func_to_fit, mass, T_min, T_max)

TK = np.linspace(700,1000, 100)
eta_vhs = lp.eta_Bird_VHS(TK, vhs_b)

print("VHS Model is:")
print(vhs_b)

eta_bouledroua_normalized = visc_func_to_fit(TK) / eta_vhs

plt.style.use('seaborn-colorblind')
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(1,1,1)

ax.plot(TK, eta_bouledroua_normalized, label='Bouledroua 2005 (fit model)')

ax.set_xlabel('T / K')
ax.set_ylabel('$\eta / \eta_\mathrm{VHS}$')
plt.title('Viscosity of lithium vapor normalized to VHS model')
plt.annotate("$D_\mathrm{ref}$: " + str(vhs_b['d_ref']), xy=(0.3,0.4), xycoords='axes fraction')
plt.annotate("$\omega$: " + str(vhs_b['omega']), xy=(0.3,0.32), xycoords='axes fraction')
plt.annotate("$T_\mathrm{ref}$: " + str(vhs_b['T_ref']), xy=(0.3,0.24), xycoords='axes fraction')

ax.set_xlim([700,1000])

ax.legend(loc=4)
plt.show()
