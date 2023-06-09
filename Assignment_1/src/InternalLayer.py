# IMPORTS #
import numpy as np
from scipy.integrate import quad

# CONSTANTS #
# The universal gravitational constant in m^3 * kg^-1 * s^-2
# https://en.wikipedia.org/wiki/Gravitational_constant
UNI_GRAV = 6.6743015E-11


class InternalLayer:
    """ Parent Class to store data on an interior layer of the
    celestial object.

    """
    def __init__(self, mat_name = "Unnamed") -> None:
        self.mat_name = mat_name


class InternalLayer_1D(InternalLayer):
    """Interior Layer Class that is only variant with radius, such that `r = 0`
    starts from the centre

    The layer density can be specified as purely homogenous, linearly increasing
    or decreasing, or a custom function - i.e. `rho(r)`
    """

    def __init__(self, r_start, r_end, cp, k, nu, alpha, rho_type, **kwargs) -> None:
        self.r_bounds = (r_start, r_end)
        self.rho_type = rho_type
        self.cp = cp
        self.k = k
        self.nu = nu
        self.alpha = alpha
        self.params = kwargs

        self.rho_function = self._rho_func_generator()
        
    def _rho_func_generator(self):
        
        if self.rho_type == "constant":
            def rho_func(r):
                return self.params["const_rho"]
            
        elif self.rho_type == "linear":
            def rho_func(r):
                b = self.params["y_int"]
                m = self.params["slope"]

                return m * (r - self.r_bounds[0]) + b

        elif self.rho_type == "custom":
            return self.params["func"]

        else:
            raise Exception("Error in rho_type input string")

        return np.vectorize(rho_func)

    def get_rho(self, r):

        if not isinstance(r, np.ndarray):
            r = np.array(r)

        assert np.all((self.r_bounds[0] <= r) * (self.r_bounds[1] >= r)), \
            "Elements within input range are out of bounds"

        return self.rho_function(r)

    def get_tot_mass(self):

        # # Use of midpoint rule (Slower than scipy quad)
        # r_range = np.linspace(self.r_bounds[0],self.r_bounds[1], N)
        # dM_dr = 4 * np.pi * self.get_rho(r_range) * r_range**2
        # mass = np.sum(np.diff(r_range) * (dM_dr[:-1] + np.diff(dM_dr)/2))

        # Use of scipy quad (fast)
        mass = quad(self.get_dM_dr, self.r_bounds[0], self.r_bounds[1])

        return mass[0]

    def get_mass(self, u_bound, ):

        l_bound = self.r_bounds[0]

        if u_bound < l_bound:
            return 0.
        elif u_bound > self.r_bounds[1]:
            return self.get_tot_mass()

        # Use of scipy quad (fast)
        mass = quad(self.get_dM_dr, l_bound, u_bound)

        return mass[0]

    # g(r) = G M(r) / r**2
    def _get_g_iso(self, r):
        return UNI_GRAV * self.get_mass(r) / r ** 2

    def get_mmoi(self):

        mmoi = quad(self.get_dI_dr, self.r_bounds[0], self.r_bounds[1])

        return mmoi[0]

    # Lambda-esque function to calculate infinitesmial shell
    # element mass
    # dM / dr = 4 pi rho r ** 2
    def get_dM_dr(self, r):
        return 4 * np.pi * self.rho_function(r) * r ** 2

    def get_dI_dr(self, r):
        return (8 / 3) * np.pi * self.rho_function(r) * r ** 4


def _test_func_1():
    Earth = InternalLayer_1D(0, 6378000,
                             "constant",
                             const_rho=5515)
    print("Test: CONSTANT")
    print("Earth Mass:", Earth.get_tot_mass())
    print("Grav accel at surface:", Earth._get_g_iso(6378E3))
    print("Earth MMoI:", Earth.get_mmoi())
    print("Earth MMoI factor:",
          Earth.get_mmoi() / ((6378E3 ** 2) * Earth.get_tot_mass()))
    print()


def _test_func_2():
    Earth = InternalLayer_1D(0, 6378000,
                             "linear",
                             y_int=10000, slope=-50.3)

    print("Test: LINEAR")
    print("Earth Mass:", Earth.get_tot_mass())
    print("Grav accel at surface:", Earth._get_g_iso(6378E3))
    print("Earth MMoI:", Earth.get_mmoi())
    print("Earth MMoI factor:",
          Earth.get_mmoi() / ((6378E3 ** 2) * Earth.get_tot_mass()))
    print()


def _test_func_3():
    def density(r):
        return 1000 - 4 * r - r ** 2.5 - 3 * r ** 1.4

    Earth = InternalLayer_1D(0, 6378000,
                             "custom",
                             func=density)

    print("Test: CUSTOM")
    print("Earth Mass:", Earth.get_tot_mass())
    print("Grav accel at surface:", Earth._get_g_iso(6378E3))
    print("Earth MMoI:", Earth.get_mmoi())
    print("Earth MMoI factor:",
          Earth.get_mmoi() / ((6378E3 ** 2) * Earth.get_tot_mass()))
    print()


if __name__ == '__main__':
    _test_func_1()
    _test_func_2()
    _test_func_3()
