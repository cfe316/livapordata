import numpy as np
import matplotlib.pyplot as plt

from lithdata.vaporpressure import *

t1 = np.linspace(1057, 2156, 60)
p1 = press_Browning_and_Potter(t1)

t2 = np.linspace(298.14, 1599.99, 60)
p2 = press_NIST_webbook(t2)

fig, ax = plt.subplots()
ax.plot(t1, p1, label='Browning and Potter, 1985')
ax.plot(t2, p2, label='NIST webbook, Antoine Equation fit')
ax.set_yscale('log')
ax.legend()
ax.set_xlim([1000,1200])
ax.set_ylim([1e2,3e3])
plt.show()

