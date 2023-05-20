from burnman import Composition
from burnman.tools.chemistry import formula_mass
import numpy as np
import matplotlib.pyplot as plt
import burnman
from burnman import Mineral, PerplexMaterial, Composite, Layer, Planet
from burnman import minerals

# Compositions from midpoints of Hirose et al. (2021), ignoring carbon and hydrogen
inner_core_composition = Composition({'Fe': 75.4, 'Ni': 9., 'Si': 10.55, 'O': 5.05}, 'weight')
outer_core_composition = Composition({'Fe': 90., 'Ni': 5., 'Si': 2., 'O': 3.}, 'weight')


for c in [inner_core_composition, outer_core_composition]:
    c.renormalize('atomic', 'total', 1.)

inner_core_elemental_composition = dict(inner_core_composition.atomic_composition)
outer_core_elemental_composition = dict(outer_core_composition.atomic_composition)
inner_core_molar_mass = formula_mass(inner_core_elemental_composition)
outer_core_molar_mass = formula_mass(outer_core_elemental_composition)

icb_radius = 400.e3
inner_core = Layer('inner core', radii=np.linspace(0., icb_radius, 50))

hcp_iron = minerals.SE_2015.hcp_iron()
params = hcp_iron.params

params['name'] = 'modified solid iron'
params['formula'] = inner_core_elemental_composition
params['molar_mass'] = inner_core_molar_mass
delta_V = 2.0e-7

inner_core_material = Mineral(params=params,
                              )

# check that the new inner core material does what we expect:
hcp_iron.set_state(5.8e9, 2000.)
inner_core_material.set_state(5.8e9, 2000.)


inner_core.set_material(inner_core_material)

inner_core.set_temperature_mode('adiabatic',temperature_top=2000)

cmb_radius = 1800.e3
outer_core = Layer('outer core', radii=np.linspace(icb_radius, cmb_radius, 50))


olivine = minerals.SLB_2011.mg_fe_olivine(molar_fractions=[0.6,0.4])
outer_core.set_material(olivine)

outer_core.set_temperature_mode('adiabatic',temperature_top=273)



from burnman import BoundaryLayerPerturbation

lab_radius = 2400.e3 # 200 km thick lithosphere
lab_temperature = 1350.

convecting_mantle_radii = np.linspace(cmb_radius, lab_radius, 101)
convecting_mantle = Layer('convecting mantle', radii=convecting_mantle_radii)

# Import a low resolution PerpleX data table.

olivine = minerals.SLB_2011.orthopyroxene(molar_fractions=[0.1,0.1,0.7,0.1])
olivine_water = Composite([olivine,minerals.HP_2011_ds62.h2oL()],fractions=[0.1,0.9])
convecting_mantle.set_material(olivine_water)

# Here we add a thermal boundary layer perturbation, assuming that the
# lower mantle has a Rayleigh number of 1.e7, and that the basal thermal
# boundary layer has a temperature jump of 840 K and the top
# boundary layer has a temperature jump of 60 K.
tbl_perturbation = BoundaryLayerPerturbation(radius_bottom=cmb_radius,
                                             radius_top=lab_radius,
                                             rayleigh_number=1.e5,
                                             temperature_change=1300.,
                                             boundary_layer_ratio=60./900.)


convecting_mantle.set_temperature_mode('perturbed-adiabatic',
                                       temperatures=tbl_perturbation.temperature(convecting_mantle_radii))



planet_radius = 2634.e3
surface_temperature = 100.
water = minerals.HP_2011_ds62.h2oL()
crust = Layer('crust', radii=np.linspace(lab_radius, planet_radius, 120))
crust.set_material(water)
crust.set_temperature_mode(temperature_mode='adiabatic',temperature_top=273)

planet_zog = Planet('Planet Zog',
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


# Now we delete the newly-created files. If you want them, comment out these lines.

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

# Finally, let's overlay some geotherms onto our model
# geotherm
labels = ['Stacey (1977)',
          'Brown and Shankland (1981)',
          'Anderson (1982)',
          'Alfe et al. (2007)',
          'Anzellini et al. (2013)']

short_labels = ['S1977',
                'BS1981',
                'A1982',
                'A2007',
                'A2013']


for i in range(2):
    ax[i].set_xticklabels([])
for i in range(4):
    ax[i].set_xlim(0., max(planet_zog.radii) / 1.e3)
    ax[i].set_ylim(0., maxy[i])

fig.set_tight_layout(True)
plt.show()