import numpy as np
from InternalLayer import *
import matplotlib.pyplot as plt
import fdm_funcs

# CONSTANTS #
# The universal gravitational constant in m^3 * kg^-1 * s^-2
# https://en.wikipedia.org/wiki/Gravitational_constant
UNI_GRAV = 6.6743015E-11


def rk4(dydx, x0, y0, x_f, steps):
    # Count number of iterations using step size or
    # step height h
    hist = [y0]
    h = abs(((x_f - x0) / steps))

    # Iterate for number of iterations
    y = y0
    for i in range(1, steps):
        "Apply Runge Kutta Formulas to find next value of y"
        k1 = h * dydx(x0, y)
        k2 = h * dydx(x0 + 0.5 * h, y + 0.5 * k1)
        k3 = h * dydx(x0 + 0.5 * h, y + 0.5 * k2)
        k4 = h * dydx(x0 + h, y + k3)

        # Update next value of y
        y = y + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        hist.append(y)
        # Update next value of x
        x0 = x0 + h
    return hist


def f_euler(dydx, x0, y0, x_f, steps):
    # Count number of iterations using step size or
    # step height h
    hist = [y0]
    h = abs(((x_f - x0) / steps))
    # Iterate for number of iterations
    y = y0
    for i in range(1, steps):
        "Apply forward difference to find next value of y"
        # Update next value of y
        y = y + h * dydx(x0, y)

        hist.append(y)
        # Update next value of x
        x0 = x0 + h
    return hist


class Celestial:
    """ Parent (Super) Class to represent celestial objects
    and calculate internal properties and stuff

    """

    def __init__(self, layers=dict(),
                 name="", ) -> None:

        self.name = name
        self.layers = layers
        self.top_bound = 0

    def _rho_func(self, r):

        for i in self.layers.values():
            if i.r_bounds[0] <= r < i.r_bounds[1]:
                return i.get_rho(r)

        return 0

    def add_layer(self, layer_name, layer):
        self.layers[layer_name] = layer
        self.top_bound = layer.r_bounds[1]

    def get_tot_mass(self,
                     rho = None, r_range = None):
        
        if rho is not None:
            M = np.cumsum(4*np.pi*rho[:-1]*r_range[:-1]**2 * np.diff(r_range))
            
            return np.append(M,M[-1])
        
        surface_r = self.layers.values()[-1].r_bounds[1]
        return self.get_mass(surface_r)

    def get_mass(self, r,):
        assert np.all(r >= 0), "distance is negative!"
        return np.sum([x.get_mass(r) for x in self.layers.values()])

    def get_g(self, r,
              rho_g = None, r_range_g = None):
        
        if rho_g is not None:
            return UNI_GRAV * self.get_tot_mass(rho = rho_g,r_range = r_range_g) / r_range_g ** 2

        assert np.all(r >= 0), "distance is negative!"
        return UNI_GRAV * self.get_mass(r) / r ** 2

    def get_mmoi(self):
        return np.sum([x.get_mmoi() for x in self.layers.values()])

    def get_p_range(self, r_probe, steps,
                    rho=None, r_range = None):

        if rho is not None:
           diff_r = np.diff(r_range)
           ghost_r = diff_r[-1]
           p_discrete = np.cumsum(rho * self.get_g(1,rho_g=rho,r_range_g=r_range) * \
                np.append(diff_r,ghost_r))
           
           return p_discrete[::-1]
           
        def p_func(r, t):
            return self._rho_func(r) * self.get_g(r)

        p_range = f_euler(p_func, r_probe, 0, self.top_bound, steps)
        # print(p_range)
        return p_range[::-1]

    def get_density(self, T, initial_rho):
        alpha_t = 3e-5
        pass
        # while 1:
        #     delta_p =
        #     delta_t = fdm_funcs()
        #     k =
        #     rho = initial_rho * (1 - alpha_t * delta_t + 1 / k * delta_p)
        #
        #
        # return density

    def _get_new_rho(self,rho,alpha_t,p, K, T_dist):
        
        return rho * (1-alpha_t*np.diff(T_dist) * (1/K) * np.diff(p))

    def get_K(self,r_range, iterations):
        converge = []
        final_k = []
        rho = np.array([self._rho_func(r) for r in r_range])
        p = np.array(self.get_p_range(r_range[0],len(r_range)))
        K = np.ones(len(r_range)) * 4E11 # Dummy K value
        alpha_t = 3E-5 # Dummy Expansion coeff
        for i in range(iterations):
            if i % 50 == 0:
                print("Iteration at:", i, "of", iterations)
            

            T_dist = fdm_funcs.diffusion_1d_steady(0, 500, [self.top_bound],
                                                [4], rho, [400], len(r_range))[1]
            
            new_rho = self._get_new_rho(rho,alpha_t,
                                        np.append(p,p[-1]),
                                        K,
                                        np.append(T_dist,T_dist[-1]))

            dp_drho = np.diff(p)/np.diff(new_rho)
            dp_drho = np.append(dp_drho,dp_drho[-1])
            
            K = rho * dp_drho
            # print(K)
            
            p = self.get_p_range(1,1,new_rho,r_range)
            rho = new_rho
            converge.append(np.sum(K)/np.max(K))
        return K, converge

# TESTS DEFINITION ##

def test_func_1():
    earth = Celestial(name="Earth")

    earth.add_layer("f1", InternalLayer_1D(0, 6378000 / 4,
                                           "constant",
                                           y_int=0, slope=5515 / (6378000 / 4 - 0),
                                           const_rho=5515,
                                           func=lambda x: 4 ** x))

    earth.add_layer("f2", InternalLayer_1D(6378000 / 4, 6378000 / 2,
                                           "constant",
                                           y_int=0, slope=5515 / (6378000 - 6378000 * 0.9999),
                                           const_rho=5515,
                                           func=lambda x: 4 ** x))

    earth.add_layer("f3", InternalLayer_1D(6378000 / 2, 6378000,
                                           "constant",
                                           y_int=5515, slope=5515 / (6378000 - 6378000 / 2),
                                           const_rho=5515,
                                           func=lambda x: 4 ** x))

    # earth.add_layer("f2", InternalLayer_1D(6378000/2, 6378000,
    #                         "linear",
    #                         y_int = 5515, slope = 0.001,
    #                         const_rho = 3000,
    #                         func = lambda x: 4**x))

    # mmoi = earth.get_mmoi()
    r_range = np.linspace(0.1, 6378000-1, 1000).tolist()

    # g_range = [earth.get_g(r) for r in r_range]
    # r_range_2 = np.linspace(6378000/2, 6378000, 1000).tolist()
    # p_ran = earth.get_p_range(6378000/2, 1000)

    rho_range = np.array([earth._rho_func(r) for r in r_range])
    r_range = np.array(r_range)
    # mass_range = [earth.get_mass(r) for r in r_range]
    # plt.plot(r_range,rho_range)

    # print("mmoi factor:", mmoi/(mass_range[-1] * earth.top_bound**2))
    # plt.plot(r_range_2, p_ran)
    # # plt.plot(r_range_2,p_ran)
    # plt.show()

    iter = 5000
    K_vals,conv = earth.get_K(r_range,iter)
    print(len(K_vals))
    # plt.plot(r_range,earth.get_g(1,rho_range,r_range))
    # plt.plot(r_range,earth.get_tot_mass(rho_range,r_range))
    # plt.plot(r_range,earth.get_p_range(1,1,rho_range,r_range))

    plt.plot(r_range,K_vals)
    plt.show()
    plt.plot([i for i in range(iter)], conv)
    plt.show()

if __name__ == '__main__':
    # TEST RUNNING #
    test_func_1()
