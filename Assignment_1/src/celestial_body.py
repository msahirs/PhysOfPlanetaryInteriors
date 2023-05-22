import numpy as np

from InternalLayer import *
from fdm_funcs import diffusion_1d_steady
import matplotlib.pyplot as plt
from utils import convergence_criteria, get_mask, apply_mask

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
            g = self.get_g(1, rho_g=rho, r_range_g=r_range)
            dp = rho * g * np.append(diff_r, diff_r[-1])
            p_discrete = np.cumsum(dp[::-1])

            return p_discrete[::-1], g

        def p_func(r, t):
            return self._rho_func(r) * self.get_g(r)

        p_range = f_euler(p_func, r_probe, 0, self.top_bound, steps)

        return p_range[::-1], None

    def get_dp_drho(self, r, rho):
        diff_r = np.diff(rho)
        return np.cumsum(r * self.get_g(1, rho_g=rho, r_range_g=r) * \
                         np.append(diff_r, diff_r[-1]))[::-1]

    def get_radii(self):
        return [_.r_bounds[1] for _ in self.layers]

    def get_cp(self):
        return [_.cp for _ in self.layers]

    def get_k(self):
        return [_.k for _ in self.layers]

    def get_rhos(self, x_grid):
        return np.array([self._rho_func(r) for r in x_grid])

    def _get_new_rho(self, rho, p, K, t, alpha_t=3E-5):
        return rho * (1 - alpha_t * t + p / K)

    def get_rayleigh(self, rho, alpha, g, t, d, kappa, nu, r_range):
        out = alpha * g * kappa
        for idx, depth in enumerate(d):
            if idx:
                mask = np.bitwise_and(d[idx - 1] < r_range, r_range <= depth)
                out[mask] *= (depth - d[idx - 1]) ** 3 / nu[idx] * (t[mask][0] - t[mask][-1])
            else:
                mask = r_range <= depth
                out[mask] *= depth ** 3 / nu[idx] * (t[mask][0] - t[mask][-1])

        return out

    def get_adiabatic_profile(self, g, cp, alpha, rho):
        return alpha * g / cp / rho

    def get_bulk_modulus(self, prev_rho, curr_rho, prev_p, curr_p):
        return prev_rho * (curr_p - prev_p) / (curr_rho - prev_rho)

    def get_alpha(self):
        return [_.alpha for _ in self.layers]

    def get_nu(self):
        return [_.nu for _ in self.layers]

    def run_convergence(self, initial_Ks, r_range, max_iterations, T=(15, 3500), epsilon=1e-5):

        # PARAMS
        steps = len(r_range)

        # VARIABLES
        l = self.get_radii()
        # get heat capacity per layer
        cp = self.get_cp()
        # get material conductivity coefficient
        k = self.get_k()
        # get thermal expansivity per layer
        alpha = self.get_alpha()
        # get viscosity per layer
        nu = self.get_nu()

        # precompute part of kappa for efficiency
        mask = get_mask(l, r_range)
        kappa = apply_mask(parameter=np.zeros(steps), mask=mask, input=[k_i / cp[idx] for idx, k_i in enumerate(k)])
        alphas = apply_mask(parameter=np.zeros(steps), mask=mask, input=alpha)
        cps = apply_mask(parameter=np.zeros(steps), mask=mask, input=cp)

        # get initial rho, p, and k
        # core to ice
        rho = [self.get_rhos(x_grid=r_range)]
        # core to ice
        K = [apply_mask(parameter=np.zeros(steps), mask=mask, input=initial_Ks)]
        # core to ice
        p = [np.array(self.get_p_range(r_range[0], steps)[0])]
        rayleigh = np.zeros_like(kappa)
        adiabatic_profile = np.zeros_like(kappa)
        Ts = []

        for i in range(max_iterations):
            if i % 5 == 0:
                print("Iteration {} / {}".format(i, max_iterations))

            _, t = diffusion_1d_steady(T=T, kappa=np.copy(kappa), rho=rho[-1], x_grid=r_range, rayleigh=rayleigh,
                                       adiabatic_profile=adiabatic_profile)
            Ts.append(t)

            # .862e+11
            rho.append(self._get_new_rho(rho=rho[0], K=.862e+11, t=t - 288.15, p=p[-1] - 101325))

            new_p, g = self.get_p_range(1, 1, rho[-1], r_range)
            p.append(new_p)

            K.append(self.get_bulk_modulus(curr_rho=rho[-1], prev_rho=rho[0], curr_p=p[-1], prev_p=p[0]))

            rayleigh = self.get_rayleigh(rho[-1], alpha=alphas, g=g, t=t, d=l, kappa=kappa, nu=nu, r_range=r_range)
            adiabatic_profile = self.get_adiabatic_profile(g=g, cp=cps, alpha=alphas, rho=rho[-1])

            if convergence_criteria([K, p, rho], epsilon):
                break

        mmoi = self.get_mmoi()
        m = self.get_tot_mass(rho=rho[-1], r_range=r_range)
        g = self.get_g(1, rho_g=rho[-1], r_range_g=r_range)

        return K, p, rho, Ts, mmoi, m, g


def run_test(rho, alpha, T, name):
    ganymede = Celestial(name=name, layers=[])

    ganymede.add_layer("Core", InternalLayer_1D(-1, 651000,
                                                rho_type="constant",
                                                y_int=0, slope=5515 / (6378000 / 4 - 0),
                                                const_rho=rho,
                                                func=lambda x: 4 ** x,
                                                cp=800,
                                                k=4,
                                                nu=2.6e-3,
                                                alpha=alpha))  # citation needed

    ganymede.add_layer("Mantle", InternalLayer_1D(651000, 1982000,
                                                  rho_type="constant",
                                                  y_int=0, slope=5515 / (6378000 - 6378000 * 0.9999),
                                                  const_rho=3100,
                                                  func=lambda x: 4 ** x,
                                                  cp=1149,
                                                  k=4,
                                                  nu=10e19,
                                                  alpha=3e-5))

    T_ice = 200  # reference: https://solarsystem.nasa.gov/moons/jupiter-moons/ganymede/in-depth/
    cp_ice = 7.49 * T_ice + 90  # reference: https://iopscience.iop.org/article/10.3847/PSJ/abcbf4/pdf
    k_ice = 567 / T_ice  # reference: https://iopscience.iop.org/article/10.3847/PSJ/abcbf4#psjabcbf4s4
    ganymede.add_layer("Ice", InternalLayer_1D(1982000, 2631000,
                                               rho_type="constant",
                                               y_int=5515, slope=5515 / (6378000 - 6378000 / 2),
                                               const_rho=1000,
                                               func=lambda x: 4 ** x,
                                               cp=cp_ice,
                                               k=k_ice,
                                               nu=10e12,
                                               alpha=30e-6))

    initial_Ks = (5.54e11, 3.91e11, 8.13e10)  # our sheet
    r_range = np.linspace(0.1, 2631000 - 1, 5000)
    return ganymede.run_convergence(initial_Ks=initial_Ks,
                                    T=(T, T_ice),
                                    r_range=r_range,
                                    max_iterations=100,
                                    epsilon=1e-5)


def run_test2(alpha, T, name):
    ganymede = Celestial(name=name, layers=[])

    ganymede.add_layer("Core", InternalLayer_1D(-1, 700000,
                                                rho_type="constant",
                                                y_int=0, slope=5515 / (6378000 / 4 - 0),
                                                const_rho=6500,
                                                func=lambda x: 4 ** x,
                                                cp=800,
                                                k=32,
                                                nu=2.6e-3,
                                                alpha=alpha))  # citation needed

    ganymede.add_layer("Mantle", InternalLayer_1D(700000, 1720000,
                                                  rho_type="constant",
                                                  y_int=0, slope=5515 / (6378000 - 6378000 * 0.9999),
                                                  const_rho=3300,
                                                  func=lambda x: 4 ** x,
                                                  cp=1149,
                                                  k=3.5,
                                                  nu=10e19,
                                                  alpha=3e-5))

    T_ice = 100  # reference: https://solarsystem.nasa.gov/moons/jupiter-moons/ganymede/in-depth/
    k_ice = 567 / T_ice  # reference: https://iopscience.iop.org/article/10.3847/PSJ/abcbf4#psjabcbf4s4
    ganymede.add_layer("Ice", InternalLayer_1D(1720000, 2634000,
                                               rho_type="constant",
                                               y_int=5515, slope=5515 / (6378000 - 6378000 / 2),
                                               const_rho=1200,
                                               func=lambda x: 4 ** x,
                                               cp=1800,
                                               k=k_ice,
                                               nu=10e12,
                                               alpha=30e-6))

    initial_Ks = (5.54e11, 3.91e11, 8.13e10)  # our sheet
    r_range = np.linspace(0.1, 2631000 - 1, 5000)
    return ganymede.run_convergence(initial_Ks=initial_Ks,
                                    T=(T, T_ice),
                                    r_range=r_range,
                                    max_iterations=100,
                                    epsilon=1e-5)


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
    # TEST RUNNING #
    K_fe, p_fe, rho_fe, T_fe, mmoi_fe, m_fe, g_fe = run_test(rho=7020, alpha=9.2e-5, T=1325, name="name")

    K_fes, p_fes, rho_fes, T_fes, mmoi_fes, m_fes, g_fes = run_test(rho=5333, alpha=1.1e-4, T=1980, name="name")

    r_range = np.linspace(0.1, 2631000 - 1, 5000)

    plt.figure()
    ax = plt.subplot(321)
    plt.plot(r_range, K_fes[-1])
    plt.plot(r_range, K_fe[-1])
    ax.axvspan(0, 651000, facecolor='grey', alpha=0.2)
    ax.axvspan(651000, 1982000, facecolor='yellow', alpha=0.2)
    ax.axvspan(1982000, 2631000, facecolor='blue', alpha=0.2)
    plt.title("Bulk Modulus")

    ax = plt.subplot(322)
    plt.plot(r_range, p_fes[-1])
    plt.plot(r_range, p_fe[-1])
    ax.axvspan(0, 651000, facecolor='grey', alpha=0.2)
    ax.axvspan(651000, 1982000, facecolor='yellow', alpha=0.2)
    ax.axvspan(1982000, 2631000, facecolor='blue', alpha=0.2)
    plt.title("Pressure")

    ax = plt.subplot(323)
    plt.plot(r_range, rho_fes[-1])
    plt.plot(r_range, rho_fe[-1])
    ax.axvspan(0, 651000, facecolor='grey', alpha=0.2)
    ax.axvspan(651000, 1982000, facecolor='yellow', alpha=0.2)
    ax.axvspan(1982000, 2631000, facecolor='blue', alpha=0.2)
    plt.title("Density")

    ax = plt.subplot(324)
    plt.plot(r_range, T_fes[-1])
    plt.plot(r_range, T_fe[-1])
    ax.axvspan(0, 651000, facecolor='grey', alpha=0.2)
    ax.axvspan(651000, 1982000, facecolor='yellow', alpha=0.2)
    ax.axvspan(1982000, 2631000, facecolor='blue', alpha=0.2)
    plt.title("Temperature")

    ax = plt.subplot(325)
    plt.plot(r_range, g_fes)
    plt.plot(r_range, g_fe)
    ax.axvspan(0, 651000, facecolor='grey', alpha=0.2)
    ax.axvspan(651000, 1982000, facecolor='yellow', alpha=0.2)
    ax.axvspan(1982000, 2631000, facecolor='blue', alpha=0.2)
    plt.title("Gravity")

    ax = plt.subplot(326)
    plt.plot(r_range, m_fes)
    plt.plot(r_range, m_fe)
    ax.axvspan(0, 651000, facecolor='grey', alpha=0.2)
    ax.axvspan(651000, 1982000, facecolor='yellow', alpha=0.2)
    ax.axvspan(1982000, 2631000, facecolor='blue', alpha=0.2)
    plt.title("Mass")

    plt.legend(["FeS", "Fe"])
    plt.tight_layout()
    plt.show()

    print("MMOI: Fe{} / FEs{}".format(mmoi_fes / m_fes[-1] / 2631000 ** 2, mmoi_fe / m_fe[-1] / 2631000 ** 2))


if __name__ == '__main__':
    # TEST RUNNING #
    K_fe, p_fe, rho_fe, T_fe, mmoi_fe, m_fe, g_fe = run_test2(alpha=9.2e-5, T=1325, name="name")

    K_fes, p_fes, rho_fes, T_fes, mmoi_fes, m_fes, g_fes = run_test2(alpha=1.1e-4, T=1980, name="name")

    r_range = np.linspace(0.1, 2631000 - 1, 5000)

    plt.figure()
    ax = plt.subplot(321)
    plt.plot(r_range, K_fes[-1])
    plt.plot(r_range, K_fe[-1])
    ax.axvspan(0, 651000, facecolor='grey', alpha=0.2)
    ax.axvspan(651000, 1982000, facecolor='yellow', alpha=0.2)
    ax.axvspan(1982000, 2631000, facecolor='blue', alpha=0.2)
    plt.title("Bulk Modulus")

    ax = plt.subplot(322)
    plt.plot(r_range, p_fes[-1])
    plt.plot(r_range, p_fe[-1])
    ax.axvspan(0, 651000, facecolor='grey', alpha=0.2)
    ax.axvspan(651000, 1982000, facecolor='yellow', alpha=0.2)
    ax.axvspan(1982000, 2631000, facecolor='blue', alpha=0.2)
    plt.title("Pressure")

    ax = plt.subplot(323)
    plt.plot(r_range, rho_fes[-1])
    plt.plot(r_range, rho_fe[-1])
    ax.axvspan(0, 651000, facecolor='grey', alpha=0.2)
    ax.axvspan(651000, 1982000, facecolor='yellow', alpha=0.2)
    ax.axvspan(1982000, 2631000, facecolor='blue', alpha=0.2)
    plt.title("Density")

    ax = plt.subplot(324)
    plt.plot(r_range, T_fes[-1])
    plt.plot(r_range, T_fe[-1])
    ax.axvspan(0, 651000, facecolor='grey', alpha=0.2)
    ax.axvspan(651000, 1982000, facecolor='yellow', alpha=0.2)
    ax.axvspan(1982000, 2631000, facecolor='blue', alpha=0.2)
    plt.title("Temperature")

    ax = plt.subplot(325)
    plt.plot(r_range, g_fes)
    plt.plot(r_range, g_fe)
    ax.axvspan(0, 651000, facecolor='grey', alpha=0.2)
    ax.axvspan(651000, 1982000, facecolor='yellow', alpha=0.2)
    ax.axvspan(1982000, 2631000, facecolor='blue', alpha=0.2)
    plt.title("Gravity")

    ax = plt.subplot(326)
    plt.plot(r_range, m_fes)
    plt.plot(r_range, m_fe)
    ax.axvspan(0, 651000, facecolor='grey', alpha=0.2)
    ax.axvspan(651000, 1982000, facecolor='yellow', alpha=0.2)
    ax.axvspan(1982000, 2631000, facecolor='blue', alpha=0.2)
    plt.title("Mass")

    plt.legend(["FeS", "Fe"])
    plt.tight_layout()
    plt.show()

    print("MMOI: Fe{} / FEs{}".format(mmoi_fes / m_fes[-1] / 2631000 ** 2, mmoi_fe / m_fe[-1] / 2631000 ** 2))
