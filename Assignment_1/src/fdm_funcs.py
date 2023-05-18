import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

""" Collection of Finite Difference functions to solve for values,
such as temperature, potential, etc.
Do try to separate steady-state and transient formulations to increase
speed and reduce computational complexity - in case of iterative usage

"""


def diffusion_1d_steady(T, kappa, rho, x_grid, ):
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
    kappa *= rho

    # Compute length of discrete element
    dx = 1 / x_steps

    # Pre-factor for elements within Finite-Difference (FD) matrix
    fd_factor = kappa / dx ** 2

    # Allocate matrix and vector for solving
    A = np.eye(x_steps)
    b = np.zeros(x_steps)

    # Setup of FD matrix
    np.fill_diagonal(A[1:-1, 1:-1], fd_factor[2:] + fd_factor[:-2])
    np.fill_diagonal(A[1:-1, 2:], -fd_factor[2:])
    np.fill_diagonal(A[1:-1, :-2], -fd_factor[:-2])

    # # Uncomment below to visualise A matrix sparsity
    # plt.spy(A, marker = "o", markersize= 15, alpha = 1, color = 'red')
    # plt.show()

    # Set up conditions of b vector for adiabatic boundary
    b[0] = T[0]
    b[-1] = T[1]

    # Making any element, except 0 and -1, non-zero represents an input
    # heat flux. This can be used to add sources/sinks to the poisson equation
    # Below line is set such that no added/removed heat
    # COMMENT: any reason? Is already 0
    # b[1:-1] = 0

    # Solve System of Equations
    sol_x = linalg.solve(A, b)

    return x_grid, sol_x


def test_func_1():
    steps = 100
    L = (900, 1400, 1900)
    x_grid = np.linspace(0, max(L), steps)
    x, y = diffusion_1d_steady(T=(15, 3700), L=L, k=(6, 5, 3), rho=np.ones(steps),
                               Cp=(570, 600, 400), x_grid=x_grid)

    plt.plot(x, y)
    plt.xlabel("Length along medium [m]")
    plt.ylabel(r"Temperature [$^\circ$C]")

    plt.show()


if __name__ == '__main__':
    test_func_1()
