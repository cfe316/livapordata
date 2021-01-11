import numpy as np
import matplotlib.pyplot as plt

from livapordata.vaporpressure import *
from livapordata.utility import error_bands

# Simple plot of all the literature vapor pressures.
# Because of the extreme ranges involved it's difficult to see the differences.

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(1,1,1)

colors = []
for i in range(10):
    color=next(ax._get_lines.prop_cycler)['color']
    colors.append(color)

tsteps = 200
t1 = np.linspace(1057, 2156, tsteps)
f1 = press_Browning_and_Potter
p1 = f1(t1)
t1_extra = np.linspace(500,2156, tsteps)
p1_extra = f1(t1_extra)

t2 = np.linspace(298.14, 1599.99, tsteps)
f2 = press_NIST_webbook
p2 = f2(t2)

t3 = np.linspace(800, 1800, tsteps)
f3 = press_Davison_1968
p3 = f3(t3)
p3_minus, p3_plus = error_bands(p3, 3.6)

t4 = np.linspace(735, 915, tsteps)
f4 = press_Maucherat_1939
p4 = f4(t4)

t5 = np.linspace(1216, 1415, tsteps)
f5 = press_Yargin_and_Sidorov
p5 = f5(t5)
t5_extra = np.linspace(516, 1216, tsteps)
p5_extra = f5(t5_extra)

t6 = np.linspace(453, 1573, tsteps)
f6 = press_Golubchikov
p6 = f6(t6)

t7 = np.linspace(700, 2000, tsteps)
f7 = press_Bystrov
p7 = f7(t7)
p7_minus, p7_plus = error_bands(p7, 2.0)

t8 = np.linspace(450, 1600, tsteps)
f8 = press_JSME_data_book
p8 = f8(t8)

t9 = np.linspace(453, 1000, tsteps)
f9 = press_Alcock
p9 = f9(t9)
p9_minus, p9_plus = error_bands(p9, 5.0)

t10 = np.linspace(1374, 1881, tsteps)
f10 = press_Bohdansky
p10 = f10(t10)

ax.plot(t1, p1, label='Browning and Potter, 1985', color=colors[0])
ax.plot(t1_extra, p1_extra, color=colors[0], dashes=[4,4])
ax.plot(t2, p2, label='NIST, fit to Hicks, 1963', color=colors[1])
ax.plot(t3, p3, label='Davison, 1968', color=colors[2])
ax.fill_between(t3, p3_minus, p3_plus, alpha=0.2, color=colors[2])

ax.plot(t4, p4, label='Maucherat, 1939', color=colors[3])
ax.plot(t5, p5, label='Yargin and Sidorov, 1982', color=colors[4])
ax.plot(t5_extra, p5_extra, color=colors[4], dashes=[4,4])
ax.plot(t6, p6, label='in Golubchikov, 1996', color=colors[5])

ax.plot(t7, p7, label='Bystrov, 1982', color=colors[6])
ax.fill_between(t7, p7_minus, p7_plus, alpha=0.2, color=colors[6])
ax.plot(t8, p8, label='JSME, 2009', color=colors[7])
ax.plot(t9, p9, label='Alcock, 1984', color=colors[8])
ax.fill_between(t9, p9_minus, p9_plus, alpha=0.2, color=colors[8])

ax.plot(t10, p10, label='Bohdansky, 1967', color=colors[9])

ax.set_yscale('log')
ax.legend()
ax.set_title('Literature vapor pressures, as reported in:')
ax.set_ylim([1e-10,3e6])
ax.set_ylabel('Pressure / Pa')
ax.set_xlabel('Temperature / K')
plt.show()

