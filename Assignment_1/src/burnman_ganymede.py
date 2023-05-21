from burnman import Composition
from burnman.tools.chemistry import formula_mass
import numpy as np
import matplotlib.pyplot as plt
import burnman
from burnman import Mineral, PerplexMaterial, Composite, Layer, Planet
from burnman import minerals

# Compositions from midpoints of Hirose et al. (2021), ignoring carbon and hydrogen
inner_core_composition = Composition({'Fe': 73.4, 'Ni': 9., 'Si': 12.55, 'O': 5.05}, 'weight')
outer_core_composition = Composition({'Fe': 90., 'Ni': 5., 'Si': 2., 'O': 3.}, 'weight')


for c in [inner_core_composition, outer_core_composition]:
    c.renormalize('atomic', 'total', 1.)

inner_core_elemental_composition = dict(inner_core_composition.atomic_composition)
outer_core_elemental_composition = dict(outer_core_composition.atomic_composition)
inner_core_molar_mass = formula_mass(inner_core_elemental_composition)
outer_core_molar_mass = formula_mass(outer_core_elemental_composition)

icb_radius = 500.e3
inner_core = Layer('inner core', radii=np.linspace(0., icb_radius, 50))

hcp_iron = minerals.SE_2015.hcp_iron()
params = hcp_iron.params

params['name'] = 'modified solid iron'
params['formula'] = inner_core_elemental_composition
params['molar_mass'] = inner_core_molar_mass

inner_core_material = Mineral(params=params,
                              )
olivine_iron = minerals.SLB_2011.mg_fe_olivine(molar_fractions=[0.5,0.5])

hcp_iron.set_state(5.8e9, 2000.)
inner_core_material.set_state(5.8e9, 2000.)


inner_core.set_material(inner_core_material)

inner_core.set_temperature_mode('adiabatic',temperature_top=2000)

cmb_radius = 1900.e3
water = minerals.HP_2011_ds62.h2oL()
outer_core = Layer('outer core', radii=np.linspace(icb_radius, cmb_radius, 100))


olivine = minerals.SLB_2011.mg_fe_olivine(molar_fractions=[0.8,0.2])
# olivine_plus_water = Composite([olivine,water],fractions=[0.6,0.4])
outer_core.set_material(olivine)

outer_core.set_temperature_mode('adiabatic',temperature_top=1800)


from burnman import BoundaryLayerPerturbation

lab_radius = 2450.e3 
# lab_temperature = 1350.

convecting_mantle_radii = np.linspace(cmb_radius, lab_radius, 100)
convecting_mantle = Layer('convecting mantle', radii=convecting_mantle_radii)

pyroxene = minerals.SLB_2011.orthopyroxene(molar_fractions=[0.1,0.3,0.3,0.3])
olivine_water = Composite([olivine,water],fractions=[0.1,0.9])
convecting_mantle.set_material(olivine_water)

tbl_perturbation = BoundaryLayerPerturbation(radius_bottom=cmb_radius,
                                             radius_top=lab_radius,
                                             rayleigh_number=1.e8,
                                             temperature_change=1500.,
                                             boundary_layer_ratio=100/900.)


convecting_mantle.set_temperature_mode('perturbed-adiabatic',
                                       temperatures=tbl_perturbation.temperature(convecting_mantle_radii))


planet_radius = 2600.e3
surface_temperature = 700.

crust = Layer('crust', radii=np.linspace(lab_radius, planet_radius, 50))

crust_ice_pyroxene = Composite([pyroxene,water],fractions=[0.01,0.99])

crust.set_material(crust_ice_pyroxene)
crust.set_temperature_mode(temperature_mode='adiabatic',temperature_top=surface_temperature)
# crust.set_pressure_mode(pressure_mode='self-consistent',pressure_top=0,gravity_bottom=1.428)
planet_zog = Planet('Ganymede',
                    [inner_core, outer_core ,
                     convecting_mantle,
                     crust], verbose=True)

planet_zog.make()

earth_mass = 1482e20
earth_moment_of_inertia_factor = 0.3105

print(f'mass = {planet_zog.mass:.3e} (ganymede = {earth_mass:.3e})')
print(f'moment of inertia factor= {planet_zog.moment_of_inertia_factor:.4f} '
      f'(ganymede = {earth_moment_of_inertia_factor:.4f})')

print('Layer mass fractions:')
for layer in planet_zog.layers:
    print(f'{layer.name}: {layer.mass / planet_zog.mass:.3f}')


fig = plt.figure(figsize=(8, 5))
ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]


bounds = np.array([[layer.radii[0]/1.e3, layer.radii[-1]/1.e3]
                   for layer in planet_zog.layers])
maxy = [15, 100, 12, 7000]
for bound in bounds:
    for i in range(4):
        ax[i].fill_betweenx([0., maxy[i]],
                            [bound[0], bound[0]],
                            [bound[1], bound[1]], alpha=0.2)

ax[0].plot(planet_zog.radii / 1.e3, planet_zog.density / 1.e3,
           label=planet_zog.name)

ax[0].set_ylabel('Density ($10^3$ kg/m$^3$)')
ax[0].legend()

# Make a subplot showing the calculated pressure profile
ax[1].plot(planet_zog.radii / 1.e3, planet_zog.pressure / 1.e9)

ax[1].set_ylabel('Pressure (GPa)')

# Make a subplot showing the calculated gravity profile
ax[2].plot(planet_zog.radii / 1.e3, planet_zog.gravity)

ax[2].set_ylabel('Gravity (m/s$^2)$')
ax[2].set_xlabel('Radius (km)')

# Make a subplot showing the calculated temperature profile
ax[3].plot(planet_zog.radii / 1.e3, planet_zog.temperature)
ax[3].set_ylabel('Temperature (K)')
ax[3].set_xlabel('Radius (km)')
ax[3].set_ylim(0.,)


for i in range(2):
    ax[i].set_xticklabels([])
for i in range(4):
    ax[i].set_xlim(0., max(planet_zog.radii) / 1.e3)
    ax[i].set_ylim(0., maxy[i])

fig.set_tight_layout(True)
plt.show()