import numpy as np
from InternalLayer import *
from fdm_funcs import diffusion_1d_steady
import matplotlib.pyplot as plt

# CONSTANTS #
# The universal gravitational constant in m^3 * kg^-1 * s^-2
# https://en.wikipedia.org/wiki/Gravitational_constant
UNI_GRAV = 6.6743015E-11


# TODO visualization, test convergence, matrix supports convection
# TODO write report
# TODO Burnman tutorial


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
        y = y + h * dydx(x0, y)

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

        for _ in self.layers:
            if _.r_bounds[0] < r <= _.r_bounds[1]:
                return _.get_rho(r)

        return 0

    def add_layer(self, layer_name, layer):
        self.layers.append(layer)
        self.top_bound = layer.r_bounds[1]

    def get_tot_mass(self,
                     rho=None, r_range=None):

        if rho is not None:
            M = np.cumsum(4 * np.pi * rho[:-1] * r_range[:-1] ** 2 * np.diff(r_range))

            return np.append(M, M[-1])

        surface_r = self.layers[-1].r_bounds[1]
        return self.get_mass(surface_r)

    def get_mass(self, r, ):
        assert np.all(r >= 0), "distance is negative!"
        return np.sum([_.get_mass(r) for _ in self.layers])

    def get_g(self, r,
              rho_g=None, r_range_g=None):

        if rho_g is not None:
            return UNI_GRAV * self.get_tot_mass(rho=rho_g, r_range=r_range_g) / r_range_g ** 2

        assert np.all(r >= 0), "distance is negative!"
        return UNI_GRAV * self.get_mass(r) / r ** 2

    def get_mmoi(self):
        return np.sum([_.get_mmoi() for _ in self.layers])

    def get_p_range(self, r_probe, steps,
                    rho=None, r_range=None):

        if rho is not None:
            diff_r = np.diff(r_range)
            dp = rho * self.get_g(1, rho_g=rho, r_range_g=r_range) * \
                                   np.append(diff_r, diff_r[-1])
            p_discrete = np.cumsum(dp[::-1])

            return p_discrete[::-1]

        def p_func(r, t):
            return self._rho_func(r) * self.get_g(r)

        p_range = f_euler(p_func, r_probe, 0, self.top_bound, steps)

        return p_range[::-1]

    def get_dp_drho(self, r, rho):
        diff_r = np.diff(rho)
        return np.cumsum(r * self.get_g(1, rho_g=rho, r_range_g=r) * \
                                   np.append(diff_r, diff_r[-1]))[::-1]

    def get_radii(self):
        return [_.r_bounds[1] for _ in self.layers]

    def get_Cp(self):
        return [_.cp for _ in self.layers]

    def get_k(self):
        return [_.k for _ in self.layers]

    def get_rhos(self, x_grid):
        return np.array([self._rho_func(r) for r in x_grid])

    def _get_new_rho(self, rho, p, K, t, alpha_t=3E-5):
        return rho * (1 - alpha_t * np.diff(t, append=1e-5) + 1 / K * np.diff(p, append=1e-5))

    def get_rayleigh(self, rho, alpha, g, delta_t, d, kappa, nu):
        return rho * alpha * g * delta_t * d **3 / kappa / nu

    def run_convergence(self, initial_rho, initial_Ks, r_range, max_iterations, T=(15, 3500), epsilon=1e-5):

        # PARAMS
        steps = len(r_range)

        # VARIABLES
        # extracting list containing radii of celestial
        L = self.get_radii()
        # get heat capacity per layer
        Cp = self.get_Cp()
        # get material conductivity coefficient
        k = self.get_k()

        # precompute part of kappa for efficiency
        kappa = np.zeros(steps)
        K = np.zeros(steps)
        initial_rhos = np.zeros(steps)

        for idx, length in enumerate(L):
            if idx:
                kappa[np.bitwise_and(L[idx - 1] < r_range, r_range <= length)] = k[idx] / Cp[idx]
                K[np.bitwise_and(L[idx - 1] < r_range, r_range <= length)] = initial_Ks[idx]
                initial_rhos[np.bitwise_and(L[idx - 1] < r_range, r_range <= length)] = initial_rho[idx]
            else:
                kappa[r_range <= length] = k[idx] / Cp[idx]
                K[r_range <= length] = initial_Ks[idx]
                initial_rhos[r_range <= length] = initial_rho[idx]

        # get initial rho, p, and k
        # core to ice
        rho = [self.get_rhos(x_grid=r_range)]
        # core to ice
        K = [K]
        # core to ice
        p = [np.array(self.get_p_range(r_range[0], steps))]

        #ts = [diffusion_1d_steady(T=T, kappa=np.copy(kappa), rho=rho[-1], x_grid=r_range)[1]]

        for i in range(max_iterations):
            if i % 5 == 0:
                print("Iteration {} / {}".format(i, max_iterations))

            _, t = diffusion_1d_steady(T=T, kappa=np.copy(kappa), rho=rho[-1], x_grid=r_range)

            #rho.append(self._get_new_rho(rho=rho[-1], K=.862e+11, t=t - 288.15, p=p[-1] - 101325))
            rho.append(self._get_new_rho(rho=rho[-1], K=.862e+11, t=t, p=p[-1]))

            #dp_drho = np.diff(p[-1]) / np.diff(rho[-1])
            #dp_drho = np.append(dp_drho, dp_drho[-1])

            #K.append(K[-1])#rho[-1] * dp_drho)

            p.append(self.get_p_range(1, 1, rho[-1], r_range))

            if convergence_criteria(K, p, rho, epsilon):
                break

        return np.array(K), np.array(p), np.array(rho)


def convergence_criteria(K: list, p: list, rho: list, epsilon):
    # if any(1 - abs(K[-1] / K[-2]) > epsilon):
    #     return False
    # print("K has converged")

    if any(1 - abs(p[-1] / p[-2]) > epsilon):
        return False
    print("p has converged")

    if any(1 - abs(rho[-1] / rho[-2]) > epsilon):
        return False
    print("rho has converged")

    return True


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
    r_range = np.linspace(0.1, 6378000 - 1, 5000).tolist()

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

    iter = 20
    K_vals, conv = earth.get_K(r_range, iter)
    print(len(K_vals))
    # plt.plot(r_range,earth.get_g(1,rho_range,r_range))
    # plt.plot(r_range,earth.get_tot_mass(rho_range,r_range))
    # plt.plot(r_range,earth.get_p_range(1,1,rho_range,r_range))

    plt.plot(r_range, K_vals)
    plt.show()
    plt.plot([i for i in range(iter)], conv)
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

    initial_rhos = (6000, 2800, 1000)  # our sheet
    initial_Ks = (5.54e11, 3.91e11, 8.13e10)  # our sheet
    r_range = np.linspace(0.1, 2634 - 1, 5000)
    K, p, rho = ganymede.run_convergence(initial_rho=initial_rhos, initial_Ks=initial_Ks, T=(1980, T_ice),
                                         r_range=r_range,
                                         max_iterations=100, epsilon=1e-5)

    plt.figure()
    plt.subplot(311)
    plt.matshow(K)
    plt.title("Bulk Modulus")

    plt.subplot(312)
    plt.matshow(p)
    plt.title("Pressure")

    plt.subplot(313)
    plt.matshow(rho)
    plt.title("Density")
    plt.tight_layout()


if __name__ == '__main__':
    # TEST RUNNING #
    test_func_2()
