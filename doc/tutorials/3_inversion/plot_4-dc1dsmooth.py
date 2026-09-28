#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
VES inversion untuk data lapangan Geolistrik 1D
================================================

Data menggunakan konfigurasi Schlumberger.

Input:
    MN/2  : jarak elektroda M-N / 2 (meter)
    AB/2  : jarak elektroda A-B / 2 (meter)
    K     : faktor geometri
    V     : tegangan (Volt)
    I     : arus (Ampere)

Perhitungan:
    R     = V / I
    rho_a = K * R

Kemudian dilakukan inversi 1D menggunakan pyGIMLi
dengan model smoothness-constrained (Occam).
"""

# %%%
# Import library
#

import numpy as np
import matplotlib.pyplot as plt

import pygimli as pg
from pygimli.physics.ves import VESRhoModelling
from pygimli.viewer.mpl import drawModel1D


# %%%
# DATA HASIL PENGUKURAN LAPANGAN
#

# MN/2 (meter)
mn2 = np.array([
    0.50,
    0.50,
    0.50,
    0.50,
    0.50,
    0.50,
    0.50,
    1.00,
    1.00,
    1.00,
    1.00,
    1.00,
    1.00,
    5.00,
    5.00,
    5.00,
    5.00,
    10.00,
    10.00
])


# AB/2 (meter)
ab2 = np.array([
    1.50,
    2.00,
    2.50,
    3.00,
    4.00,
    5.00,
    6.00,
    6.00,
    7.00,
    8.00,
    9.00,
    10.00,
    12.00,
    15.00,
    20.00,
    25.00,
    30.00,
    30.00,
    40.00
])


# Faktor geometri K
K = np.array([
    6.29,
    11.79,
    18.86,
    27.50,
    49.50,
    77.79,
    112.36,
    55.00,
    75.43,
    99.00,
    125.71,
    155.57,
    224.71,
    352.00,
    62.86,
    117.86,
    188.57,
    125.71,
    235.71
])


# Tegangan V (Volt)
V = np.array([
    0.319,
    0.222,
    0.162,
    0.125,
    0.104,
    0.102,
    0.101,
    0.021,
    0.026,
    0.004,
    0.007,
    0.011,
    0.011,
    0.011,
    0.070,
    0.020,
    0.019,
    0.043,
    0.015
])


# Arus I (Ampere)
I = np.array([
    0.097,
    0.108,
    0.107,
    0.093,
    0.076,
    0.086,
    0.105,
    0.105,
    0.100,
    0.101,
    0.067,
    0.112,
    0.114,
    0.115,
    0.084,
    0.057,
    0.088,
    0.088,
    0.093
])


# %%%
# Hitung resistansi dan resistivitas semu
#

# Resistansi
R = V / I

# Resistivitas semu
rhoa = K * R


# %%%
# Tampilkan data hasil perhitungan
#

print("\n==============================================")
print(" DATA RESISTIVITAS SEMU")
print("==============================================")

print(
    "{:>8} {:>8} {:>10} {:>10} {:>10} {:>12} {:>15}".format(
        "MN/2", "AB/2", "K", "V", "I", "R (Ohm)",
        "Rho_a (Ohm.m)"
    )
)

for i in range(len(ab2)):

    print(
        "{:8.2f} {:8.2f} {:10.2f} {:10.3f} {:10.3f} "
        "{:12.4f} {:15.4f}".format(
            mn2[i],
            ab2[i],
            K[i],
            V[i],
            I[i],
            R[i],
            rhoa[i]
        )
    )


# %%%
# Tampilkan kurva data lapangan
#

plt.figure(figsize=(7, 6))

plt.loglog(
   (ab2,
    rhoa,
    'x-',
    label='Data lapangan'
)

plt.xlabel('AB/2 [m]')
plt.ylabel(r'$\rho_a$ [$\Omega$m]')
plt.title('Kurva Resistivitas Semu - Schlumberger')

plt.grid(True, which='both')
plt.legend()

plt.show()


# %%%
# Tentukan error data
#
# Karena data lapangan tidak mempunyai nilai error pengukuran
# pada tabel, digunakan error relatif 5%.
#
# Nilai ini dapat diganti jika nilai error alat sudah diketahui.
#

error = np.ones(len(rhoa)) * 0.05


# %%%
# Set up forward operator
#
# Model terdiri dari beberapa lapisan dengan ketebalan
# yang dibuat secara smooth.
#

thk = np.logspace(-0.5, 0.5, 30)

f = VESRhoModelling(
    thk=thk,
    ab2=ab2,
    mn2=mn2
)


# %%%
# Set up inversion
#

inv = pg.Inversion(
    fop=f,
    verbose=False
)


# %%%
# Transformasi data dan model
#

# Transformasi log untuk data
inv.transData = pg.trans.TransLog()

# Resistivitas dibatasi antara 1 sampai 10000 Ohm.m
inv.transModel = pg.trans.TransLogLU(
    1,
    10000
)


# %%%
# Inversi dengan regularisasi lambda = 100
#

print("\nInversion dengan lam = 100")

res100 = inv.run(
    rhoa,
    error,
    lam=100
)

print(
    'rrms = {:.2f}%, chi^2 = {:.3f}'.format(
        inv.relrms(),
        inv.chi2()
    )
)


# %%%
# Inversi dengan regularisasi lambda = 10
#

print("\nInversion dengan lam = 10")

res10 = inv.run(
    rhoa,
    error,
    lam=10
)

print(
    'rrms = {:.2f}%, chi^2 = {:.3f}'.format(
        inv.relrms(),
        inv.chi2()
    )
)


# %%%
# Inversi dengan second order smoothness
#

print("\nInversion dengan second order smoothness")

inv.setRegularization(cType=2)

resC2 = inv.run(
    rhoa,
    error,
    lam=20
)

print(
    'rrms = {:.2f}%, chi^2 = {:.3f}'.format(
        inv.relrms(),
        inv.chi2()
    )
)


# %%%
# Visualisasi model resistivitas 1D
#

fig, ax = plt.subplots(
    ncols=2,
    figsize=(9, 6)
)


# ---------------------------------------------
# Grafik model lapisan
# ---------------------------------------------

drawModel1D(
    ax[0],
    thk,
    res100,
    color='C1',
    label=r'$\lambda$ = 100',
    plot='semilogx'
)

drawModel1D(
    ax[0],
    thk,
    res10,
    color='C2',
    label=r'$\lambda$ = 10'
)

drawModel1D(
    ax[0],
    thk,
    resC2,
    color='C4',
    label=r'C = 2'
)

ax[0].grid(True, which='both')

ax[0].set_xlabel(
    r'$\rho$ [$\Omega$m]'
)

ax[0].set_ylabel(
    'Depth [m]'
)

ax[0].set_title(
    'Model Resistivitas 1D'
)

ax[0].legend(
    loc='best'
)


# ---------------------------------------------
# Grafik data lapangan dan hasil fitting
# ---------------------------------------------

ax[1].loglog(
    rhoa,
    ab2,
    'x-',
    color='C0',
    label='Measured'
)

ax[1].loglog(
    inv.response,
    ab2,
    '-',
    color='C5',
    label='Fitted'
)

ax[1].set_ylim(
    (
        max(ab2),
        min(ab2)
    )
)

ax[1].grid(
    True,
    which='both'
)

ax[1].set_xlabel(
    r'$\rho_a$ [$\Omega$m]'
)

ax[1].set_ylabel(
    'AB/2 [m]'
)

ax[1].yaxis.set_label_position(
    'right'
)

ax[1].yaxis.set_ticks_position(
    'right'
)

ax[1].legend(
    loc='best'
)

ax[1].set_title(
    'Data Lapangan vs Hasil Inversi'
)

plt.tight_layout()
plt.show()
