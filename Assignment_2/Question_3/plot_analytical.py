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
lamb_low = 2 * 1500E3   # Loading wavelength (2*crater diam)
lamb_high = 2 * 750E3   # Loading wavelength (2*crater diam)
w_low = 1960    # Inital depression
w_high = 1897

# Get values
t_r_low = time_const(rho_cr, nu, g_surf,lamb_low)
t_range_low = np.linspace(0,6*t_r_low,50)
w_range_low = exp_decrement(t_range_low,t_r_low,w_low)

print("Time constant (low): %.3E" % t_r_low)
print(f"Time Interval for analytical (low): {t_range_low[1]:.3E}", )

t_r_high = time_const(rho_cr, nu, g_surf,lamb_high)
t_range_high = np.linspace(0,6*t_r_high,50)
w_range_high = exp_decrement(t_range_high,t_r_high,w_high)

print("Time constant (high): %.3E" % t_r_high)
print(f"Time Interval for analytical (high): {t_range_high[1]:.3E}", )


# ABAQUS OUTPUT LISTS

abs_mins_low = [1960,1477,1173,943.1,766,627.4,517.6,429.8]

abs_mins_high = [1897,1162,801.7,571.5,419.8,317.7,247.8,199.5]

t_stamps = [0] + list(accumulate([2.0E13]*7))


# PLOTTING

# ABAQUS calculation
plt.plot(t_stamps, abs_mins_low,
         'x--', color = 'blue', alpha = 0.7,
         label = "ABAQUS Model (low)")

plt.plot(t_stamps, abs_mins_high,
         'x--', color = 'orange', alpha = 0.7,
         label = "ABAQUS Model (high)")

# Analytical calculation (low)
plt.plot(t_range_low, w_range_low,
         color = 'blue', alpha = 0.7, 
         label = f"Turcotte and Schubert, \n $t_r = {t_r_low:.3E}$s")

# plt.plot(t_range_high, w_range_high,
#          color = 'orange', alpha = 0.7, 
#          label = f"Turcotte and Schubert, \n $\lambda = {lamb_high:.3E}$m , $t_r = {t_r_high:.3E}$s")


plt.title("Numerical (ABAQUS) vs Analytical Solutions")
plt.xlabel("Elapsed time [t]")
plt.ylabel("Subsurface displacement [m]")
plt.legend()

plt.show()