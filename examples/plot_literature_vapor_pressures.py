import numpy as np
import matplotlib.pyplot as plt

from lithdata.vaporpressure import *
from lithdata.utility import error_bands

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(1,1,1)

colors = []
for i in range(5):
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

ax.plot(t1, p1, label='Browning and Potter, 1985', color=colors[0])
ax.plot(t2, p2, label='NIST webbook, Antoine Equation fit', color=colors[1])
ax.plot(t3, p3, label='Davison, 1968', color=colors[2])
plt.fill_between(t3, p3_minus, p3_plus, alpha=0.2, color=colors[2])

ax.plot(t4, p4, label='Maucherat, 1939', color=colors[3])

ax.set_yscale('log')
ax.legend()
#ax.set_xlim([1000,1200])
#ax.set_ylim([1e2,3e3])
plt.show()

