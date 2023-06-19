import matplotlib.pyplot as plt
import numpy as np
from itertools import accumulate

#Get time constant
def time_const(rho_cr,nu,g_surf,lamb):

    # Abaqus input param b_diff (see slides)
    def b_diff(nu):
        return 1/(3*nu)

    return (4 * np.pi * b_diff(nu)) / (rho_cr * g_surf * lamb)

# exponential decay formula
def exp_decrement(t,t_r,initial_disp):
    return initial_disp * np.exp(-t/t_r)

# SCRIPT START

#Input params of crust and loading

rho_cr = 3000   # Crustal Density
nu = 3.333E-24  # Mantle Visocsity
g_surf = 1.9144     # Surface gravitational strength
lamb = 2 * 1000E3   # Loading wavelength (2*crater diam)
w = 1813    # Inital depression

# Get values
t_r = time_const(rho_cr, nu, g_surf,lamb)
t_range = np.linspace(0,10*t_r,50)
w_range = exp_decrement(t_range,t_r,w)

print("Time constant: %.3E" % t_r)
print(f"Time Interval for analytical: {t_range[1]:.3E}", )


# ABAQUS OUTPUT LISTS
abs_mins = [1813, 1264,946.4,720.8,556.5,
            434.7,343.3,274.0,221.1,180.6,
            151.1,129.3,112.5,99.03,88.58,
            79.95,72.77,66.72,61.6,57.28,
            53.55,50.28]
t_stamps = [0] + list(accumulate([2.0E13]*21))


# PLOTTING

# ABAQUS calculation
plt.plot(t_stamps, abs_mins,
         'x--', color = 'blue', alpha = 0.7,
         label = "ABAQUS Model")

# Analytical calculation
plt.plot(t_range, w_range,
         color = 'orange', alpha = 0.7, 
         label = f"Turcotte and Schubert, \n $t_r = {t_r:.3E} $")


plt.title("Numerical (ABAQUS) vs Analytical Solutions")
plt.xlabel("Elapsed time [t]")
plt.ylabel("Subsurface displacement [m]")
plt.legend()

plt.show()