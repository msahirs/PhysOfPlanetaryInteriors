import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

""" Collection of Finite Difference functions to solve for values,
such as temperature, potential, etc.
Do try to separate steady-state and transient formulations to increase
speed and reduce computational complexity - in case of iterative usage

"""

def diffusion_1d_steady(T_start, T_end, L, k, rho, Cp, x_steps, ):
    """`diffusion_1d_steady(...)` is used to calculate the 1D
    Poisson (Heat) equation at steady state.

    - I basically solve this https://en.wikipedia.org/wiki/Numerical_solution_of_the_convection-diffusion_equation
    but in 1D and with the time derivative terms removed (steady-state). The boundary conditions are set to be adiabatic

    - https://people.sc.fsu.edu/~jburkardt/classes/math1091_2020/poisson_1d_steady/poisson_1d_steady.pdf also gives an
    example of the discretisation and numerical methods used

    Parameters
    ----------
    `T_start` : float
        Fixed Temperature at start of medium
    `T_end` : float
        Fixed Temperature at end of medium
    `L` : float
        Length of 1-D medium
    `k` : float
        Material conductivity coeffcient
    `rho` : float
        Material mass density
    `Cp` : float
        Specific Heat capacity
    `x_steps` : int
        Number of interior nodes

    Returns
    -------
   `x_grid` : ndarray
        Length-wise discretisation of points
    `sol_x` : ndarray
        Steady-state temperature at discrete nodes
    """    

    #Calculation of Diffusion Coefficient (see description links)
    a =  k/(Cp * rho)

    #Discretise medium into x_steps number of nodes
    x_grid = np.linspace(0,L,x_steps)
    
    #Compute length of discrete element
    dx = 1/x_steps
    
    #Pre-factor for elements within Finite-Difference (FD) matrix
    fd_factor = a/dx**2

    #Allocate matrix and vector for solving
    A = np.zeros((x_steps,x_steps))
    b = np.zeros((x_steps,1)) 

    #Setup of FD matrix
    np.fill_diagonal(A,2*fd_factor)
    np.fill_diagonal(A[:-1,1:],-fd_factor)
    np.fill_diagonal(A[1:, :-1],-fd_factor)

    #Change first and last row entries for adiabatic boundary
    #conditions
    A[0,:] = 0
    A[-1,:] = 0
    A[0,0] = 1
    A[-1,-1] = 1

    # # Uncomment below to visualise A matrix sparsity
    # plt.spy(A, marker = "o", markersize= 15, alpha = 1, color = 'red')
    # plt.show()

    #Set up conditions of b vector for adiabatic boundary
    b[0] = T_start
    b[-1] = T_end

    # Making any element, except 0 and -1, non-zero represents an input
    # heat flux. This can be used to add sources/sinks to the poisson equation
    # Below line is set such that no added/removed heat
    b[1:-1] = 0
    
    #Solve System of Equations
    sol_x = linalg.solve(A,b)

    return x_grid, sol_x

def test_func_1():
    x, y = diffusion_1d_steady(15,3700,2900000,4,3500,570,10)

    plt.plot(x,y)
    plt.xlabel("Length along medium [m]")
    plt.ylabel(r"Temperature [$^\circ$C]")

    plt.show()

test_func_1()
