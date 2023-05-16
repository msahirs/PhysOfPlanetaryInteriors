import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

def diffusion_1d_steady(T_start, T_end, L, k, rho, Cp, x_steps, ):
    
    a =  k/(Cp * rho)
    x_grid = np.linspace(0,L,x_steps)
    dx = 1/x_steps
    fd_factor = a/dx**2
    print(fd_factor)

    A = np.zeros((x_steps,x_steps))
    
    b = np.zeros((x_steps,1)) 

    np.fill_diagonal(A,2*fd_factor)
    np.fill_diagonal(A[:-1,1:],-fd_factor)
    np.fill_diagonal(A[1:, :-1],-fd_factor)

    A[0,:] = 0
    A[-1,:] = 0
    A[0,0] = 1
    A[-1,-1] = 1
    # # plt.spy(A, marker = "o", markersize= 15, alpha = 1, color = 'red')
    # # print(A)
    b[0] = T_start
    b[1:-1] = 0.05
    b[-1] = T_end
    # b[10:-50] = 0
    # print(A)
    print(b)
    sol_x = linalg.solve(A,b)
    print(sol_x)
    # sol_x = np.insert(sol_x,0,T_start)    
    # sol_x = np.append(sol_x,T_end)
    plt.plot(x_grid,sol_x)
    plt.show()
    return 0

diffusion_1d_steady(15,3700,2900000,4,3500,570,10)