import numpy as np
from InternalLayer import *
# CONSTANTS #
from matplotlib import pyplot as plt

# The universal gravitational constant in m^3 * kg^-1 * s^-2
# https://en.wikipedia.org/wiki/Gravitational_constant
UNI_GRAV = 6.6743015E-11

def rk4(dydx, x0, y0, x_f, steps):
    # Count number of iterations using step size or
    # step height h
    hist = [y0]
    h = abs(((x_f - x0)/steps))

    # Iterate for number of iterations
    y = y0
    for i in range(1, steps):
        "Apply Runge Kutta Formulas to find next value of y"
        k1 = h * dydx(x0, y)
        k2 = h * dydx(x0 + 0.5 * h, y + 0.5 * k1)
        k3 = h * dydx(x0 + 0.5 * h, y + 0.5 * k2)
        k4 = h * dydx(x0 + h, y + k3)
 
        # Update next value of y
        y = y + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)
        hist.append(y)
        # Update next value of x
        x0 = x0 + h
    return hist


class Celestial:
    """ Parent (Super) Class to represent celestial objects
    and calculate internal properties and stuff

    """        
    def __init__(self, layers = dict(),
                 name = "",) -> None:
        
        self.name = name
        self.layers = layers
        self.top_bound = 0
    
    def _rho_func(self,r):

        for i in self.layers.values():
            
            if r >= i.r_bounds[0] and r < i.r_bounds[1]:
                return i.get_rho(r)

        return 0

    def add_layer(self,layer_name,layer):
        self.layers[layer_name] = layer
        self.top_bound = layer.r_bounds[1]

    def get_tot_mass(self):
        surface_r = self.layers.values()[-1].r_bounds[1]
        return self.get_mass(surface_r)
    
    def get_mass(self,r):
        assert np.all(r >= 0), "distance is negative!"
        return np.sum([x.get_mass(r) for x in self.layers.values()])
        

    def get_g(self, r):
        
        assert np.all(r >= 0), "distance is negative!"
        return UNI_GRAV * self.get_mass(r) / r**2
        

    def get_mmoi(self):
        return np.sum([x.get_mmoi() for x in self.layers.values()])
    
    def get_p_range(self,r_probe,steps):
        def p_func(r,t):
            
            return self._rho_func(r) * self.get_g(r)
        
        p_range = rk4(p_func,r_probe,0,self.top_bound,steps)
        
        return p_range[::-1]
        

    
earth = Celestial(name="Earth")

earth.add_layer("f1", InternalLayer_1D(0, 6378000/2,
                         "constant",
                         y_int = 10000, slope = -50.3,
                         const_rho = 5515,
                         func = lambda x: 4**x))

earth.add_layer("f2", InternalLayer_1D(6378000/2, 6378000,
                         "linear",
                         y_int = 5515, slope = -0.001,
                         const_rho = 3000,
                         func = lambda x: 4**x))

# mmoi = earth.get_mmoi()
# r_range = np.linspace(0.1,6378000*1.2,10000).tolist()

# g_range = [earth.get_g(r) for r in r_range]
r_range_2 = np.linspace(0.1,6378000,10000).tolist()
p_ran = earth.get_p_range(0.1,10000)



# rho_range = [earth._rho_func(r) for r in r_range]
# mass_range = [earth.get_mass(r) for r in r_range]
plt.plot(r_range_2,p_ran)
plt.show()