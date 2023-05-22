import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

""" Collection of Finite Difference functions to solve for values,
such as temperature, potential, etc.
Do try to separate steady-state and transient formulations to increase
speed and reduce computational complexity - in case of iterative usage

"""


def diffusion_1d_steady(T, kappa, rho, x_grid, rayleigh, adiabatic_profile):
    """`diffusion_1d_steady(...)` is used to calculate the 1D
    Poisson (Heat) equation at steady state.

    - I basically solve this https://en.wikipedia.org/wiki/Numerical_solution_of_the_convection-diffusion_equation
    but in 1D and with the time derivative terms removed (steady-state). The boundary conditions are set to be adiabatic

    - https://people.sc.fsu.edu/~jburkardt/classes/math1091_2020/poisson_1d_steady/poisson_1d_steady.pdf also gives an
    example of the discretisation and numerical methods used

    Parameters
    ----------
    `T` : tuple
        Fixed Temperature at core and outer of medium
    `L` : list
        radii of all layers
    `k` : list
        Material conductivity coeffcient per layer
    `rho` : list
        Material mass density per layer
    `Cp` : list
        Specific Heat capacity per layer
    `x_grid` : ndarray
        x

    Returns
    -------
   `x_grid` : ndarray
        Length-wise discretisation of points
    `sol_x` : ndarray
        Steady-state temperature at discrete nodes
    """

    # Discretise medium into x_steps number of nodes
    x_steps = len(x_grid)

    # Calculation of Diffusion Coefficient (see description links)
    kappa /= rho

    # Compute length of discrete element
    dx = (x_grid[1] - x_grid[0])

    # Pre-factor for elements within Finite-Difference (FD) matrix
    fd_factor = kappa / dx ** 2

    # Allocate matrix and vector for solving
    A = np.eye(x_steps)
    b = np.zeros(x_steps)

    fd_factor[rayleigh > 1E4] = 1 / dx / 2

    # Setup of FD matrix for conductive heating
    np.fill_diagonal(A[1:-1, 1:-1], fd_factor[2:] + fd_factor[:-2])

    fd_factor1 = -fd_factor
    fd_factor2 = -fd_factor

    np.fill_diagonal(A[1:-1, :-2], fd_factor1[:-2])
    np.fill_diagonal(A[1:-1, 2:], fd_factor2[2:])

    A[0, 0] = 1
    A[-1, -1] = 1

    # Set up conditions of b vector for adiabatic boundary
    b[rayleigh > 1E4] = adiabatic_profile[rayleigh > 1E4]
    b[0] = T[0]
    b[-1] = T[1]

    # Solve System of Equations
    sol_x = linalg.solve(A, b)

    return x_grid, sol_x


def test_func_1():
    steps = 100
    L = (900, 1400, 1900)
    x_grid = np.linspace(0, max(L), steps)
    k = [4,3,2]
    Cp = [800, 1149, 1588]
    kappa = np.zeros(steps)
    for idx, length in enumerate(L):
        if idx:
            kappa[np.bitwise_and(L[idx - 1] < x_grid, x_grid <= length)] = k[idx] / Cp[idx]
        else:
            kappa[x_grid <= length] = k[idx] / Cp[idx]
    x, y = diffusion_1d_steady(T=(1980, 200), kappa=kappa, rho=np.ones(steps)*6000, x_grid=x_grid)

    plt.plot(x, y)
    plt.xlabel("Length along medium [m]")
    plt.ylabel(r"Temperature [$^\circ$C]")

    plt.show()


if __name__ == '__main__':
    test_func_1()
