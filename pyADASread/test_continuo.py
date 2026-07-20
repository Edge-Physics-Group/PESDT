import ctypes
import os, sys
import matplotlib.pyplot as plt
import numpy as np


class ContinuumRadiation(ctypes.Structure):
    _fields_ = [
        ("free_free", ctypes.c_double),
        ("total", ctypes.c_double),
    ]


lib = ctypes.CDLL("./libcontinuo_.so")

lib.continuo_.argtypes = [
    ctypes.c_double,  # wavelength_A
    ctypes.c_double,  # Te_eV
    ctypes.c_int,     # atomic_number
    ctypes.c_int,     # ion_charge
]


'''
lib.argam.argtypes = [ctypes.c_int, ctypes.c_double]
lib.r8f21_.argtypes= [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
lib.r8fdip0_.argtypes = [ctypes.c_double, ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_double]
lib.r8fmon1_.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_int]
lib.r8fdip1_.argtypes = [ctypes.c_double, ctypes.c_int, ctypes.c_double, ctypes.c_int]
lib.r8fdip2_.argtypes = [ctypes.c_double, ctypes.c_int, ctypes.c_double, ctypes.c_int]
lib.r8fdip_.argtypes = [ctypes.c_double, ctypes.c_int, ctypes.c_double, ctypes.c_int]
lib.r8giii_.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double]
lib.r8giiiav_.argtypes = [ctypes.c_double, ctypes.c_double]
lib.r8gbf_.argtypes = [ctypes.c_double, ctypes.c_double]
lib.r8gav_.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_int]
'''
lib.continuo_.restype = ContinuumRadiation  
'''
lib.argam.restype = ctypes.c_double
lib.r8f21_.restype= ctypes.c_double
lib.r8fdip0_.restype = ctypes.c_double
lib.r8fmon1_.restype = ctypes.c_double
lib.r8fdip1_.restype = ctypes.c_double
lib.r8fdip2_.restype = ctypes.c_double
lib.r8fdip_.restype = ctypes.c_double
lib.r8giii_.restype = ctypes.c_double
lib.r8giiiav_.restype = ctypes.c_double
lib.r8gbf_.restype = ctypes.c_double
lib.r8gav_.restype = ctypes.c_double
'''
def call_continuo(wavelength_A, Te_eV, atomic_number, ion_charge):
    result = lib.continuo_(
        float(wavelength_A),
        float(Te_eV),
        int(atomic_number),
        int(ion_charge),
    )

    return result.free_free, result.total
# 4. Generate Validation Data Sweep
# Define physics scenario: Hydrogen plasma (Z0=6, Z1=5) at 20 eV temperature

Te_test = 100
Z0_test = 1
Z1_test = 1
# Sweep wavelengths from 300 Å to 500 Å
wavelengths = np.linspace(3000.0, 5000.0, 200)
ff_intensities = np.zeros_like(wavelengths)
fb_intensities = np.zeros_like(wavelengths)

for i, wl in enumerate(wavelengths):
    ff, fb = call_continuo(wl, Te_test, Z0_test, Z1_test)
    ff_intensities[i] = ff
    fb_intensities[i] = fb

# 5. Plot the results to visually inspect recombination edges
plt.figure(figsize=(10, 6))
plt.plot(
    wavelengths,
    ff_intensities,
    label="Free-Free (Bremsstrahlung)",
    color="crimson",
    linestyle="--",
)
plt.plot(
    wavelengths,
    fb_intensities,
    label="Free-Bound (Recombination)",
    color="royalblue",
)
plt.plot(
    wavelengths,
    ff_intensities + fb_intensities,
    label="Total Continuum",
    color="black"
)

plt.title(
    f"Plasma Emission Continuum Verification\n$T_e$ = {Te_test} eV, Charge State: Z={Z1_test}"
)
plt.xlabel("Wavelength (Å)")
plt.ylabel("Intensity (Arbitrary Units)")
plt.yscale("log")  # Continuum features are easiest to see on a log scale
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.show()