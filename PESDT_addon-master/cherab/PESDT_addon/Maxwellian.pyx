# cython: language_level=3



from libc.math cimport exp, M_PI
from raysect.optical cimport Vector3D
cimport cython

from cherab.core.math cimport autowrap_function3d, autowrap_vectorfunction3d
from cherab.core.utility.constants cimport ELEMENTARY_CHARGE
from cherab.core cimport DistributionFunction
from cherab.core.math cimport Function3D, VectorFunction3D

cdef class PESDTMaxwellian(DistributionFunction):
    """
    A Maxwellian distribution function.

    This class implements a Maxwell-Boltzmann distribution, the statistical distribution
    describing a system of particles that have reached thermodynamic equilibrium. The
    user supplies 3D functions that provide the mean density, temperature and velocity
    respectively.

    :param Function3D density: 3D function defining the density in cubic meters.
    :param Function3D temperature: 3D function defining the temperature in eV.
    :param VectorFunction3D velocity: 3D vector function defining the bulk velocity in meters per second.
    :param double atomic_mass: Atomic mass of the species in kg.

    """

    def __init__(self, object density, object temperature, object velocity, object emission, double atomic_mass):
        cdef :
            Function3D _density, _temperature, _emission
            VectorFunction3D _velocity
            double _atomic_mass
            dict _emission_dict
            
        super().__init__()
        self._density = autowrap_function3d(density)
        self._temperature = autowrap_function3d(temperature)
        self._velocity = autowrap_vectorfunction3d(velocity)
        self._emission_dict = emission
        self._emission = autowrap_function3d(self._emission_dict[next(iter(self._emission_dict))])
        self._atomic_mass = atomic_mass

    @cython.cdivision(True)
    cdef double evaluate(self, double x, double y, double z, double vx, double vy, double vz) except? -1e999:
        """
        Evaluates the phase space density at the specified point in 6D phase space.

        :param x: position in meters
        :param y: position in meters
        :param z: position in meters
        :param vx: velocity in meters per second
        :param vy: velocity in meters per second
        :param vz: velocity in meters per second
        :return: phase space density in s^3/m^6
        """

        cdef:
            double k1, k2, ux, uy, uz
            Vector3D bulk_velocity

        k1 = self._atomic_mass / (2 * ELEMENTARY_CHARGE * self._temperature.evaluate(x, y, z))
        k2 = (k1 / M_PI) ** 1.5

        bulk_velocity = self._velocity.evaluate(x, y, z)
        ux = vx - bulk_velocity.x
        uy = vy - bulk_velocity.y
        uz = vz - bulk_velocity.z

        return self._density.evaluate(x, y, z) * exp(-k1 * (ux*ux + uy*uy + uz*uz)) * k2

    cpdef Vector3D bulk_velocity(self, double x, double y, double z):
        """
        Evaluates the species' bulk velocity at the specified 3D coordinate.

        :param x: position in meters
        :param y: position in meters
        :param z: position in meters
        :return: velocity vector in m/s
        
        .. code-block:: pycon

           >>> d0_distribution.bulk_velocity(1, 0, 0)
           Vector3D(0.0, 0.0, 0.0)
        """

        return self._velocity.evaluate(x, y, z)

    cpdef double effective_temperature(self, double x, double y, double z) except? -1e999:
        """

        :param x: position in meters
        :param y: position in meters
        :param z: position in meters
        :return: temperature in eV
        
        .. code-block:: pycon
        
           >>> d0_distribution.effective_temperature(1, 0, 0)
           1.0
        """

        return self._temperature.evaluate(x, y, z)

    cpdef double density(self, double x, double y, double z) except? -1e999:
        """

        :param x: position in meters
        :param y: position in meters
        :param z: position in meters
        :return: density in m^-3

        .. code-block:: pycon

           >>> d0_distribution.density(1, 0, 0)
           1e+17
        """

        return self._density.evaluate(x, y, z)

    cdef double emission(self, double x, double y, double z):

        return self._emission.evaluate(x, y, z)

    cpdef void update_emission(self, transition):
        """
        Updates the emission to the current transition
        """
        self._emission = autowrap_function3d(self._emission_dict[transition])