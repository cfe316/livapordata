import numpy as np
import matplotlib.pyplot as plt
from lithdata.li_property_library import LiPropertyLibrary
from lithdata.fit_dsmc_vss import VSS_model_from_eta_and_D11
from lithdata.constants import mLi

lp = LiPropertyLibrary()

visc_func = lp.eta1_Bouledroua_Table_I
diff_func = lp.D11_Fialho_1993_Table
mass = mLi
T_min, T_max = 700, 1000
vss = VSS_model_from_eta_and_D11(visc_func, diff_func, mass, T_min, T_max)

TK = np.linspace(700,1000, 100)
eta_vss = lp.eta_Bird_VSS(TK, vss)

print("VSS Model is:")
print(vss)

eta_bouledroua_normalized = visc_func(TK) / eta_vss
eta_v_normalized = lp.eta1_Vargaftik_1991_Table(TK) / eta_vss

plt.style.use('seaborn-colorblind')
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(1,1,1)

ax.plot(TK, eta_bouledroua_normalized, label='Bouledroua 2005 (fit model)')

ax.set_xlabel('T / K')
ax.set_ylabel('$\eta / \eta_\mathrm{VHS}$')
plt.title('Viscosity of lithium vapor normalized to VSS model')
plt.annotate("$D_\mathrm{ref}$: " + str(vss['d_ref']), xy=(0.3,0.4), xycoords='axes fraction')
plt.annotate("$\omega$: " + str(vss['omega']), xy=(0.3,0.32), xycoords='axes fraction')
plt.annotate("$T_\mathrm{ref}$: " + str(vss['T_ref']), xy=(0.3,0.24), xycoords='axes fraction')
plt.annotate(r"$\alpha$: " + str(vss['alpha']), xy=(0.3,0.16), xycoords='axes fraction')

ax.set_xlim([700,1000])

ax.legend(loc=4)
plt.show()
