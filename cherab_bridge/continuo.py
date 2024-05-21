import numpy as np
from cherab.core.plasma import PlasmaModel
from raysect.optical import Spectrum, Point3D, Vector3D
from cherab.core.utility.conversion import PhotonToJ
from adaslib.atomic import continuo

class Continuo(PlasmaModel):
    """
    Emitter that calculates bremsstrahlung emission from a plasma object using the ADAS
    adaslib/continuo.f function.

    """

    def __repr__(self):
        return '<PlasmaModel - adaslib/continuo Bremsstrahlung>'

    def emission(self, point, direction, spectrum):

        ne = self.plasma.electron_distribution.density(point.x, point.y, point.z)
        if ne == 0:
            return spectrum
        te = self.plasma.electron_distribution.effective_temperature(point.x, point.y, point.z)
        if te == 0:
            return spectrum
        z_effective = self.plasma.z_effective(point.x, point.y, point.z)
        if z_effective == 0:
            return spectrum

        # numerically integrate using trapezium rule
        # todo: add sub-sampling to increase numerical accuracy
        lower_wavelength = spectrum.min_wavelength
        lower_sample = self._continuo(lower_wavelength, te, ne, zeff=z_effective)
        for i in range(spectrum.bins):

            upper_wavelength = spectrum.min_wavelength + spectrum.delta_wavelength * i


            upper_sample = self._continuo(upper_wavelength, te, ne, zeff=z_effective)

            spectrum.samples[i] += 0.5 * (lower_sample + upper_sample)

            lower_wavelength = upper_wavelength
            lower_sample = upper_sample

        return spectrum

    def _continuo(self, wvl, te, ne, zeff=1):
        # TODO: implement zeff using adaslib/continuo function. This is a weighted sum of the
        # main ion and impurity ion densities
        """
        adaslib/continuo wrapper

        :param wvl: in nm
        :param te: in eV
        :param ne: in m^-3
        :param zeff: a.u.
        :return:

        /home/adas/python/adaslib/atomic/continuo.py doc

          PURPOSE    : calculates continuum emission at a requested wavelength
                       and temperature for an element ionisation stage.

          contff, contin = continuo(wave, tev, iz0, iz1)

                        NAME         TYPE     DETAILS
          REQUIRED   :  wave()       float    wavelength required (A)
                        tev()        float    electron temperature (eV)
                        iz0          int      atomic number
                        iz1          int      ion stage + 1

          RETURNS       contff(,)    float    free-free emissivity (ph cm3 s-1 A-1)
                        contin(,)    float    total continuum emissivity
                                              (free-free + free-bound) (ph cm3 s-1 A-1)
                                                  dimensions: wave, te (dropped if just 1).

          MODIFIED   :
               1.1     Martin O'Mullane
                         - First version

          VERSION    :
                1.1    16-06-2016


        """
        wvl_A = wvl * 10.
        iz0=1
        iz1=1
        contff, contin = continuo(wvl_A, te, iz0, iz1)

        # Convert to ph/s/m^3/str/nm
        contin = (1. / (4 * np.pi)) * contin * ne * ne * (1.0e-06) * 10.0 # ph/s/m^3/str/nm

        radiance =  PhotonToJ.to(contin, wvl) # W/m^3/str/nm

        return radiance



