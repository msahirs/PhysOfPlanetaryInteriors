import matplotlib.pyplot as plt
import numpy as np

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
rho_cr = 3000
nu = 3.333E-24
g_surf = 1.9144
lamb = 2 * 1000E3
w = 4000

# Get values
t_r = time_const(rho_cr,nu,g_surf,lamb)
t_range = np.linspace(0,10*t_r,100)
w_range = exp_decrement(t_range,t_r,w)

# PLOTTING

# ABAQUS calculation
plt.plot(t_range, w_range,
         'x--', color = 'blue', label = "ABAQUS Model")

# Analytical calculation
plt.plot(t_range, w_range,
         color = 'orange', label = "Turcotte and Schubert")


plt.title("Numerical (ABAQUS) vs Analytical Solutions")
plt.xlabel("Elapsed time [t]")
plt.ylabel("Subsurface displacement [m]")
plt.legend()

plt.show()