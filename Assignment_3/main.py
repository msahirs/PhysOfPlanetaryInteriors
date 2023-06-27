import matplotlib.pyplot as plt
import pyshtools as pysh
from pyshtools import constants
from cartopy import crs as ccrs
import numpy as np
import pandas as pd

pysh.utils.figstyle(rel_width=1)


def get_data():
    data = {
        'clm': pysh.datasets.Venus.MGNP180U(),
        'shape': pysh.datasets.Venus.VenusTopo719(lmax=719),
        'u0': 0,
        'a': constants.Venus.r.value,
        'f': 0,
        'LMAX': 120,
        'th_grav': 200,
        'th_boug': 400,
        'dens_boug': 2900
    }
    return data


def plot_topology(data):
    clm = data['shape'] / 1e3  # divide by 1000 to convert to km
    clm.coeffs[0, 0, 0] = 0.  # set both the degree 0
    clm.coeffs[0, 2, 0] = 0.  # and the flattening terms to zero,
    grid = clm.expand()
    plt.show()
    grid.plot(tick_interval=[60, 45],
              minor_tick_interval=[30, 15],
              cmap_reverse=True,
              colorbar='right',
              cmap='RdBu',
              cb_label='Elevation (km)',
              title='Venus Topography',
              show=False)


def plot_gravity_disturbance(data):
    grav = data['clm'].expand(a=data['a'], f=data['f'], lmax=data['LMAX'])
    grav.plot_total(tick_interval=[60, 45],
                    minor_tick_interval=[30, 15],
                    colorbar='right',
                    cmap='RdBu_r',
                    cmap_limits=[-data['th_grav'], data['th_grav']],
                    title='Venus Gravity Disturbance',
                    show=False)
    plt.show()


def plot_bouger_correction(data):
    # create a Bourger Correction from the shape
    bc = pysh.SHGravCoeffs.from_shape(data['shape'],
                                      rho=data['dens_boug'],
                                      gm=data['clm'].gm,
                                      lmax=data['LMAX'])
    # TODO add comment
    bc = bc.change_ref(r0=data['clm'].r0)

    # TODO add comment
    bc.set_coeffs(ls=0, ms=0, values=0)
    bc.set_coeffs(ls=2, ms=0, values=0)

    # TODO add comment
    bouguer = data['clm'].pad(lmax=data['LMAX']) - bc

    # TODO add comment
    bouguer_grid = bouguer.expand(lmax=data['LMAX'], a=data['a'], f=data['f'])

    bouguer_grid.plot_total(tick_interval=[60, 45],
                            minor_tick_interval=[30, 15],
                            colorbar='right',
                            cmap='RdBu_r',
                            cmap_limits=[-data['th_boug'], data['th_boug']],
                            cb_triangles='both',
                            title='Bouguer Gravity Map',
                            show=False)
    plt.show()


def run_question2(data):
    plot_topology(data)
    plot_gravity_disturbance(data)
    plot_bouger_correction(data)


if __name__ == "__main__":
    data = get_data()
    run_question2(data)
    print("here")


