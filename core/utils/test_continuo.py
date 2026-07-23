import ctypes
import os, sys
import matplotlib.pyplot as plt
import numpy as np
from adaslib import continuo as adas_continuo
from continuo_read import continuov_, CONV

class ContinuumRadiation(ctypes.Structure):
    _fields_ = [
        ("free_free", ctypes.c_double),
        ("total", ctypes.c_double),
    ]

lib_root = os.environ.get("CONTINUO_LIB")
if lib_root is None:

    lib_path = "./libcontinuo_.so"
else:
    lib_path = os.path.join(lib_root, "libcontinuo_.so")
lib = ctypes.CDLL(lib_path)

lib.continuo_.argtypes = [
    ctypes.c_double,  # wavelength_A
    ctypes.c_double,  # Te_eV
    ctypes.c_int,     # atomic_number
    ctypes.c_int,     # ion_charge
]


lib.continuo_.restype = ContinuumRadiation  

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

Te_tests = [1, 2, 3, 5, 10, 20, 50, 100]

for Te_test in Te_tests:
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

    ffv, ffvtot = continuov_(wavelengths, Te_test, 1, 1)
    ffa, fftot = adas_continuo(wavelengths, Te_test, 1, 1)

    ffv = ffv.flatten()/CONV; ffvtot = ffvtot.flatten()/CONV
    ffa = ffa.flatten(); fftot = fftot.flatten()
    # 5. Plot the results to visually inspect recombination edges
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    ax.plot(wavelengths,ff_intensities,label="Free-Free (Bremsstrahlung)",color="crimson")
    ax.plot(wavelengths,fb_intensities,label="Free-Bound (Recombination)",color="royalblue")
    ax.plot(wavelengths,ff_intensities + fb_intensities,label="Total Continuum",color="black")

    ax.plot(wavelengths,ffv,label="FF - vectorfunc",color="green",linestyle="--")
    ax.plot(wavelengths,ffvtot + fb_intensities,label="Total vectorfunc",color="green")

    ax.plot(wavelengths,ffa,label="FF - adas",color="magenta",linestyle="--")
    ax.plot(wavelengths,fftot + fb_intensities,label="Total adas",color="magenta")

    ax2.plot(wavelengths,(ff_intensities-ffa)/ffa,label="FF -residual",color="crimson")
    ax2.plot(wavelengths,(fb_intensities-(fftot-ffa))/(fftot-ffa),label="FB -residual",color="royalblue")
    ax2.plot(wavelengths,(ff_intensities-fftot + fb_intensities)/fftot,label="Total -residual",color="black")

    ax2.plot(wavelengths,(ffv-ffa)/ffa,label="FF - vectorfunc residual",color="green",linestyle="--")
    ax2.plot(wavelengths,(ffvtot-fftot)/fftot + fb_intensities,label="Total vectorfunc residual",color="green")


    ax.set_title(f"Plasma Emission Continuum Verification\n$T_e$ = {Te_test} eV, Charge State: Z={Z1_test}")
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Intensity (Arbitrary Units)")
    ax.set_yscale("log")  # Continuum features are easiest to see on a log scale
    ax.grid(True, which="both", alpha=0.3)

    ax2.set_xlabel("Wavelength (Å)")
    ax2.set_ylabel("residual (me-adas)")

    ax.legend()
plt.show()