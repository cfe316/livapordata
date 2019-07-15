import numpy as np
import matplotlib.pyplot as plt

from lithdata.vaporpressure import *
from lithdata.utility import error_bands

fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(1,1,1)

colors = []
for i in range(6):
    color=next(ax._get_lines.prop_cycler)['color']
    colors.append(color)

# browning and potter
t1 = np.linspace(1057, 2156, 2)

# nist webbook
t2 = np.linspace(298.14, 1599.99, 2)

# Davison
t3 = np.linspace(800, 1800, 2)

# Maucherat
t4 = np.linspace(735, 915, 2)

#Vargaftik and Kapitonov
t5 = np.linspace(1216, 1415, 2)

# Golubchikov
t6 = np.linspace(453, 1573, 2)

ax.plot(t5, [6,6], label='Vargaftik and Kapitonov, 1985', color=colors[4])
ax.text(t5[0] - 50, 6, 'Vargaftik and Kapitonov, 1985', horizontalalignment='right', verticalalignment='center')
ax.plot(t1, [5,5], label='Browning and Potter, 1985', color=colors[0])
ax.text(t1[0] - 50, 5, 'Browning and Potter, 1985', horizontalalignment='right', verticalalignment='center')
ax.plot(t3, [4,4], label='Davison, 1968', color=colors[2])
ax.text(t3[0] - 50, 4, 'Davison, 1968', horizontalalignment='right', verticalalignment='center')
ax.plot(t4, [3,3], label='Maucherat, 1939', color=colors[3])
ax.text(t4[-1] + 50, 3, 'Maucherat, 1939', horizontalalignment='left', verticalalignment='center')
ax.plot(t6, [2,2], label='Golubchikov, 1996', color=colors[5])
ax.text(t6[-1] + 50, 2, 'Golubchikov, 1996', horizontalalignment='left', verticalalignment='center')
ax.plot(t2, [1,1], label='NIST webbook, Antoine Equation fit', color=colors[1])
ax.text(t2[-1] + 50, 1, 'NIST webbook (Hicks 1963)', horizontalalignment='left', verticalalignment='center')

#ax.legend()
ax.set_title('Ranges of literature vapor pressures, as reported in:')
ax.set_ylim([0.5,6.5])
ax.set_yticks([])
ax.set_xlabel('Temperature / K')
plt.show()

