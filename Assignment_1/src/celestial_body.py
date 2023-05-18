import numpy as np
from InternalLayer import *
from fdm_funcs import diffusion_1d_steady
import matplotlib.pyplot as plt

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


def f_euler(dydx, x0, x_f, y0, steps):
    # Count number of iterations using step size or
    # step height h
    hist = [y0]
    h = abs(((x_f - x0) / steps))
    # Iterate for number of iterations
    y = y0
    for i in range(1, steps):
        "Apply forward difference to find next value of y"
        # Update next value of y
        y = y + h * dydx(x0, y, i)

        hist.append(y)
        # Update next value of x
        x0 = x0 + h
    return hist


class Celestial:
    """ Parent (Super) Class to represent celestial objects
    and calculate internal properties and stuff

    """

    def __init__(self, layers=list(),
                 name="", ) -> None:

        self.name = name
        self.layers = layers
        self.top_bound = 0

    def _rho_func(self, r):

        for l in self.layers:
            if l.r_bounds[0] < r <= l.r_bounds[1]:
                return l.get_rho(r)

        return 0

    def add_layer(self, layer_name, layer):
        self.layers.append(layer)
        self.top_bound = layer.r_bounds[1]

    def get_tot_mass(self):
        surface_r = self.layers[-1].r_bounds[1]
        return self.get_mass(surface_r)

    def get_mass(self, r):
        assert np.all(r >= 0), "distance is negative!"
        return np.sum([_.get_mass(r) for _ in self.layers])

    def get_g(self, r):

        assert np.all(r >= 0), "distance is negative!"
        return UNI_GRAV * self.get_mass(r) / r ** 2

    def get_mmoi(self):
        return np.sum([_.get_mmoi() for _ in self.layers])

    def get_p_range(self, steps, rho, r_probe=1e-7):
        # pressure from r_surface to r_probe
        def p_func(r, t, i):
            return rho[i] * self.get_g(r)

        p_range = f_euler(p_func, r_probe, self.top_bound, 0, steps)

        return np.array(p_range[::-1])

    def get_radius(self):
        return [_.r_bounds[1] for _ in self.layers]

    def get_Cp(self):
        return [_.cp for _ in self.layers]

    def get_k(self):
        return [_.k for _ in self.layers]

    def get_rhos(self, x_grid):
        return np.array([self._rho_func(r) for r in x_grid])

    def get_density(self, initial_rho, initial_Ks, T, steps, r_probe=1e-7, epsilon=1e-2):

        # CONSTANT
        alpha_t = 3e-5

        # VARIABLES
        # extracting list containing radii of celestial
        L = self.get_radius()
        # get heat capacity per layer
        Cp = self.get_Cp()
        # get material conductivity coefficient
        k = self.get_k()

        x_grid = np.linspace(0, max(L), steps)

        # get initial_rho
        rho = [self.get_rhos(x_grid=x_grid)]

        # precompute part of kappa for efficiency
        kappa = np.zeros(steps)
        K = np.zeros(steps)
        initial_rhos = np.zeros(steps)

        for idx, length in enumerate(L):
            if idx:
                kappa[np.bitwise_and(L[idx - 1] < x_grid, x_grid <= length)] = k[idx] / Cp[idx]
                K[np.bitwise_and(L[idx - 1] < x_grid, x_grid <= length)] = initial_Ks[idx]
                initial_rhos[np.bitwise_and(L[idx - 1] < x_grid, x_grid <= length)] = initial_rho[idx]
            else:
                kappa[x_grid <= length] = k[idx] / Cp[idx]
                K[x_grid <= length] = initial_Ks[idx]
                initial_rhos[x_grid <= length] = initial_rho[idx]

        #while 1:
        for i in range(100):
            # compute change in pressure
            p = self.get_p_range(r_probe=r_probe, rho=rho[-1], steps=steps)

            # compute change in temperature
            _, t = diffusion_1d_steady(T, np.copy(kappa), rho[-1], x_grid)

            # compute new density
            rho.append(initial_rhos * (1 - alpha_t * np.diff(t, append=1e-7) + np.diff(p, append=1e-7) / K))

            K = np.diff(p, append=1e-7) / (np.diff(rho[-1], append=1e-7) + 1e-10)
            K *= rho[-1]
            K += 1e-10

            if all(abs(rho[-1] - rho[-2]) < epsilon):
                print("Converged")
                break

        return x_grid, np.array(rho)


# TESTS DEFINITION ##

def test_func_1():
    earth = Celestial(name="ganymede")

    earth.add_layer("f1", InternalLayer_1D(0, 6378000 / 4,
                                           "linear",
                                           y_int=0, slope=5515 / (6378000 / 4 - 0),
                                           const_rho=5515,
                                           func=lambda x: 4 ** x))

    earth.add_layer("f2", InternalLayer_1D(6378000 / 4, 6378000 / 2,
                                           "constant",
                                           y_int=0, slope=5515 / (6378000 - 6378000 * 0.9999),
                                           const_rho=5515,
                                           func=lambda x: 4 ** x))

    earth.add_layer("f3", InternalLayer_1D(6378000 / 2, 6378000,
                                           "linear",
                                           y_int=5515, slope=5515 / (6378000 - 6378000 / 2),
                                           const_rho=5515,
                                           func=lambda x: 4 ** x))

    # ganymede.add_layer("f2", InternalLayer_1D(6378000/2, 6378000,
    #                         "linear",
    #                         y_int = 5515, slope = 0.001,
    #                         const_rho = 3000,
    #                         func = lambda x: 4**x))

    # mmoi = ganymede.get_mmoi()
    r_range = np.linspace(0.1, 6378000, 10000).tolist()

    # g_range = [ganymede.get_g(r) for r in r_range]
    r_range_2 = np.linspace(0.1, 6378000, 1000).tolist()
    p_ran = earth.get_p_range(0.1, 1000)

    # rho_range = [ganymede._rho_func(r) for r in r_range]

    # mass_range = [ganymede.get_mass(r) for r in r_range]
    # plt.plot(r_range,rho_range)

    # print("mmoi factor:", mmoi/(mass_range[-1] * ganymede.top_bound**2))
    plt.plot(r_range_2, p_ran)
    # plt.plot(r_range_2,p_ran)
    plt.show()


def test_func_2():
    ganymede = Celestial(name="Ganymede")

    ganymede.add_layer("Core", InternalLayer_1D(-1, 900,
                                                rho_type="constant",
                                                y_int=0, slope=5515 / (6378000 / 4 - 0),
                                                const_rho=5150,
                                                func=lambda x: 4 ** x,
                                                cp=800,
                                                k=4))  # citation needed

    ganymede.add_layer("Mantle", InternalLayer_1D(900, 1849,
                                                  rho_type="constant",
                                                  y_int=0, slope=5515 / (6378000 - 6378000 * 0.9999),
                                                  const_rho=5515,
                                                  func=lambda x: 4 ** x,
                                                  cp=1149,
                                                  k=4))

    T_ice = 200  # reference: https://solarsystem.nasa.gov/moons/jupiter-moons/ganymede/in-depth/
    cp_ice = 7.49 * T_ice + 90  # reference: https://iopscience.iop.org/article/10.3847/PSJ/abcbf4/pdf
    k_ice = 567 / T_ice  # reference: https://iopscience.iop.org/article/10.3847/PSJ/abcbf4#psjabcbf4s4
    ganymede.add_layer("Ice", InternalLayer_1D(1849, 2634,
                                               rho_type="constant",
                                               y_int=5515, slope=5515 / (6378000 - 6378000 / 2),
                                               const_rho=1182,
                                               func=lambda x: 4 ** x,
                                               cp=cp_ice,
                                               k=k_ice))

    initial_rhos = (6000, 2800, 1000)           # our sheet
    initial_Ks = (5.54e11, 3.91e11, 8.13e10)    # our sheet
    x, y = ganymede.get_density(initial_rho=initial_rhos, initial_Ks=initial_Ks, T=(1980, T_ice), steps=100, epsilon=1e-2)

    plt.matshow(y)
    plt.show()


if __name__ == '__main__':
    # TEST RUNNING #
    test_func_2()
