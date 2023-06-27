import matplotlib.pyplot as plt
import numpy as np
from itertools import accumulate


# step wall clock times
visc_21 = [4148,1384,1407,1380]
visc_22 = [4158,1488,1410,1409]
visc_23 = [4058,1411,1398,1383]
visc_24 = [4184,1413,1397,1377]

steps = list(range(0,4))

plt.plot(steps, visc_21, 'o-',
         color = 'red', alpha = 0.7, 
         label = "Viscosity = $10^{21} Pa\,s$")

plt.plot(steps, visc_22, 'o-',
         color = 'blue', alpha = 0.7, 
         label = "Viscosity = $10^{22} Pa\,s$")

plt.plot(steps, visc_23, 'o-',
         color = 'green', alpha = 0.7, 
         label = "Viscosity = $10^{23} Pa\,s$")

plt.plot(steps, visc_24, 'o-',
         color = 'orange', alpha = 0.7, 
         label = "Viscosity = $10^{24} Pa\,s$")

plt.title("Clock time comparison with viscosity variation (1 CPU)")
plt.xlabel("Step number")
plt.xticks(steps)
plt.ylabel("Clock time [s]")
plt.legend()

plt.show()