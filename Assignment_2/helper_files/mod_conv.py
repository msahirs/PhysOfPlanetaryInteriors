"""Used to calculate Young's Modulus from Shear modulus.
Note that this is only valid for isotropic and homogenous materials
"""
import numpy as np

poisson_ratio = 0.28
shear_collec = dict(shear_core =  2E9,shear_mantle = 9E7, shear_crust = 6.5E10)

shear_mods = np.array(list(shear_collec.values()))


young_mods = 2 * shear_mods * (1 + poisson_ratio)
for i in range(young_mods.size):
    print(list(shear_collec.keys())[i],": {:.2E}".format(young_mods[i]))