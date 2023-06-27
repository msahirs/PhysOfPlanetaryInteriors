"""Used to calculate Young's Modulus from Shear modulus.
Note that this is only valid for isotropic and homogenous materials
"""
import numpy as np

visc_collec = dict(visc_core =  2E9, visc_mantle = 9E7, visc_crust = 6.5E10)

visc_collec_array = np.array(list(visc_collec.values()))

b_diffs = 1/(3*visc_collec_array)

for i in range(visc_collec_array.size):
    print(list(visc_collec.keys())[i],": {:.2E}".format(b_diffs[i]))