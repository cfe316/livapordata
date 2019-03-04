import numpy as np
import matplotlib.pyplot as plt

from lithdata.vaporpressure import *
from lithdata.utility import error_bands

# Simple plot of all the literature vapor pressures.
# Because of the extreme ranges involved it's difficult to see the differences.

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(1,1,1)

colors = []
for i in range(6):
    color=next(ax._get_lines.prop_cycler)['color']
    colors.append(color)

t1 = np.linspace(1057, 2156, 60)
p1 = press_Browning_and_Potter(t1)

t2 = np.linspace(298.14, 1599.99, 60)
p2 = press_NIST_webbook(t2)

t3 = np.linspace(800, 1800, 60)
p3 = press_Davison_1968(t3)
p3_minus, p3_plus = error_bands(p3, 3.6)

t4 = np.linspace(735, 915, 60)
p4 = press_Maucherat_1939(t4)

t5 = np.linspace(1216, 1415, 60)
p5 = press_Vargaftik_and_Kapitonov(t5)

t6 = np.linspace(453, 1573, 60)
p6 = press_Golubchikov(t6)

ax.plot(t1, p1, label='Browning and Potter, 1985', color=colors[0])
ax.plot(t2, p2, label='NIST webbook, Antoine Equation fit', color=colors[1])
ax.plot(t3, p3, label='Davison, 1968', color=colors[2])
plt.fill_between(t3, p3_minus, p3_plus, alpha=0.2, color=colors[2])

ax.plot(t4, p4, label='Maucherat, 1939', color=colors[3])
ax.plot(t5, p5, label='Vargaftik and Kapitonov, 1985', color=colors[4])
ax.plot(t6, p6, label='Golubchikov, 1996', color=colors[5])

ax.set_yscale('log')
ax.legend()
ax.set_title('Literature vapor pressures, as reported in:')
ax.set_ylim([1e-10,3e6])
ax.set_ylabel('Pressure / Pa')
ax.set_xlabel('Temperature / K')
plt.show()

