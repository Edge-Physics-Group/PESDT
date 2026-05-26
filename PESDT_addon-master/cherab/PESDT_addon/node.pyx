# Copyright 2016-2018 Euratom
# Copyright 2016-2018 United Kingdom Atomic Energy Authority
# Copyright 2016-2018 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
#
# Licensed under the EUPL, Version 1.1 or – as soon they will be approved by the
# European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
# https://joinup.ec.europa.eu/software/page/eupl5
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the Licence is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
#
# See the Licence for the specific language governing permissions and limitations
# under the Licence.

# cython: language_level=3
from cherab.core.utility import Notifier

from cherab.core.species import SpeciesNotFound
from raysect.optical cimport AffineMatrix3D, Vector3D
from raysect.core cimport Node, Primitive

from raysect.optical.material.emitter.inhomogeneous cimport VolumeIntegrator
from raysect.optical.material.emitter.inhomogeneous cimport NumericalIntegrator

from cherab.core.math cimport Function3D, autowrap_function3d
from cherab.core.math cimport VectorFunction3D, autowrap_vectorfunction3d
from cherab.core.distribution cimport DistributionFunction, ZeroDistribution
from cherab.core.plasma.material cimport PlasmaMaterial
from cherab.core.atomic cimport AtomicData, Element
from cherab.core.distribution cimport DistributionFunction
from cherab.core.species cimport Species
from cherab.core.math cimport VectorFunction3D
from cherab.core.plasma.model cimport PlasmaModel
from .spectrum cimport OpaqueSpectrum
from .PlasmaModel cimport OpaquePlasmaModel
cimport cython


cdef class Composition:
    """
    The plasma composition manager.

    Used to control the adding and removing of Species objects from the Plasma object.
    This is because there can only ever be one Species object instance for each plasma
    species of a given element and charge state. Users never instantiate this class
    directly. Its always used indirectly through an instantiated Plasma object.
    """

    def __init__(self):

        self._species = {}
        self.notifier = Notifier()

    def __len__(self):
        return len(self._species)

    def __iter__(self):
        """
        Used to iterate over all the Species objects in the parent plasma.

        .. code-block:: pycon

           >>> [species for species in plasma.composition]
           [<Species: element=deuterium, charge=0>,
            <Species: element=deuterium, charge=1>]
        """

        return iter(self._species.values())

    def __getitem__(self, tuple item):
        """
        Species objects can be indexed with a tuple specifying their element and charge state.

        .. code-block:: pycon

           >>> plasma.composition[(deuterium, 0)]
           <Species: element=deuterium, charge=0>
        """

        try:
            element, charge = item
        except ValueError:
            raise ValueError('An (element, charge) tuple is required containing the element and '
                             'charge state of the species.')
        return self.get(element, charge)

    cpdef object set(self, object species):
        """
        Replaces the species in the composition with a new list of species.

        If there are multiple species with the same element and charge state in
        the list, only the last species with that specification will be added
        to the composition.

        :param Species species: A list containing the new species.
        
        .. code-block:: pycon
        
           >>> d0_species = Species(deuterium, 0, d0_distribution)
           >>> d1_species = Species(deuterium, 1, d1_distribution)
           >>> plasma.composition.set([d0_species, d1_species])
           >>> [species for species in plasma.composition]
           [<Species: element=deuterium, charge=0>,
            <Species: element=deuterium, charge=1>]
        """

        # must be an iterable
        species = tuple(species)

        # check contents of list are species
        for item in species:
            if not isinstance(item, Species):
                raise TypeError('The composition list must consist of only Species objects.')

        # reset composition
        self._species = {}

        for item in species:
            self._species[(item.element, item.charge)] = item
        self.notifier.notify()

    cpdef object add(self, Species species):
        """
        Adds a species to the plasma composition.
        
        Replaces any existing species with the same element and charge
        state already in the composition.
        
        :param Species species: A Species object.
        
        .. code-block:: pycon
        
           >>> d1_species = Species(deuterium, 1, d1_distribution)
           >>> plasma.composition.add(d1_species)
        """

        if not species:
            raise ValueError('Species must not be None type.')

        self._species[(species.element, species.charge)] = species
        self.notifier.notify()

    cpdef Species get(self, Element element, int charge):
        """
        Get a specified plasma species.
        
        Raises a ValueError if the specified species is not found in the composition.
        
        :param Element element: The element object of the requested species.
        :param int charge: The charge state of the requested species.
        :return: The requested Species object.
        
        .. code-block:: pycon

           >>> plasma.composition.get(deuterium, 1)
           <Species: element=deuterium, charge=1>
        """

        try:
            return self._species[(element, charge)]
        except KeyError:
            raise ValueError("Could not find a species with the specified element '{}' and charge {}."
                             "".format(element.name, charge))

    cpdef object clear(self):
        """Removes all Species object instances from the parent plasma."""

        self._species = {}
        self.notifier.notify()


cdef class ModelManager:

    def __init__(self):
        self._models = []
        self.notifier = Notifier()

    def __iter__(self):
        return iter(self._models)

    cpdef object set(self, object models):

        # copy models and test it is an iterable
        models = list(models)

        # check contents of list are plasma models
        for model in models:
            if not isinstance(model, OpaquePlasmaModel):
                raise TypeError('The model list must consist of only PlasmaModel objects.')

        self._models = models
        self.notifier.notify()

    cpdef object add(self, OpaquePlasmaModel model):

        if not model:
            raise ValueError('Model must not be None type.')

        self._models.append(model)
        self.notifier.notify()

    cpdef object clear(self):
        self._models = []
        self.notifier.notify()


cdef class OpaquePlasma(Node):
    """
    An hack to get opacity done (Fuck this OO mess...)
    """

    def __init__(self, object parent=None, AffineMatrix3D transform=None, str name=None,
                 integrator=NumericalIntegrator(step=0.001)):

        super().__init__(parent, transform, name)

        # plasma modification notifier
        self.notifier = Notifier()

        # plasma properties
        self.b_field = None
        self.electron_distribution = None

        # setup plasma composition handler and pass through notifications
        self._composition = Composition()
        self._composition.notifier.add(self._modified)

        # atomic data source passed to emission models
        self._atomic_data = None

        # plasma geometry
        self._geometry = None
        self._geometry_transform = None

        # setup emission model handler and trigger geometry rebuilding if the models change
        self._models = ModelManager()
        self._models.notifier.add(self._configure_geometry)

        # emission model integrator
        self._integrator = integrator

    @property
    def b_field(self):
        return self._b_field

    @b_field.setter
    def b_field(self, object value):
        # assign Vector3D(0, 0, 0) if None is passed
        if value is None:
            self._b_field = autowrap_vectorfunction3d(Vector3D(0, 0, 0))
        else:
            self._b_field = autowrap_vectorfunction3d(value)

        self._modified()

    # cython fast access
    cdef VectorFunction3D get_b_field(self):
        return self._b_field

    @property
    def electron_distribution(self):
        return self._electron_distribution

    @electron_distribution.setter
    def electron_distribution(self, DistributionFunction value):
        # assign ZeroDistribution if None value passed
        if value is None:
            self._electron_distribution = ZeroDistribution()
        else:
            self._electron_distribution = value

        self._modified()

    # cython fast access
    cdef DistributionFunction get_electron_distribution(self):
        return self._electron_distribution

    @property
    def composition(self):
        return self._composition

    @composition.setter
    def composition(self, object values):
        self._composition.set(values)

    # cython fast access
    cdef Composition get_composition(self):
        return self._composition

    @cython.cdivision(True)
    cpdef double z_effective(self, double x, double y, double z) except -1:
        r"""
        Calculates the effective Z of the plasma.

        .. math::
            Z_{eff} = \sum_{j=1}^N n_{i(j)} Z_j^2 / \sum_{k=1}^N n_{i(k)} Z_k

        where n is the species density and Z is the ionisation of the species.

        :param x: x coordinate in meters.
        :param y: y coordinate in meters.
        :param z: z coordinate in meters.
        :return: Calculated Z effective.
        :raises ValueError: If plasma does not contain any ionised species.
        
        .. code-block:: pycon
           
           >>> # With an already initialised plasma object...
           >>> plasma.z_effective(1, 1, 1)
           1.0
        """

        cdef:
            double ion_density, sum_nz, sum_nz2
            Species species

        sum_nz = 0
        sum_nz2 = 0
        for species in self._composition:
            if species.charge > 0:
                density = species.distribution.density(x, y, z)
                sum_nz += density * species.charge
                sum_nz2 += density * species.charge * species.charge

        if sum_nz2 == 0:
            raise ValueError('Plasma does not contain any ionised species.')

        return sum_nz2 / sum_nz

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef double ion_density(self, double x, double y, double z):
        r"""
        Calculates the total ion density of the plasma.

        .. math::
            n_I = \sum_{k=1}^N n_i(k)

        :param x: x coordinate in meters.
        :param y: y coordinate in meters.
        :param z: z coordinate in meters.
        :return: Total ion density in m^-3.
        
        .. code-block:: pycon
           
           >>> # With an already initialised plasma object...
           >>> plasma.ion_density(1, 1, 1)
           1.1e+18
        """

        cdef:
            double ion_density = 0.0
            Species species

        for species in self._composition:
            ion_density += species.distribution.density(x, y, z)
        return ion_density

    @property
    def geometry(self):
        return self._geometry

    @geometry.setter
    def geometry(self, Primitive value):
        self._geometry = value
        self._configure_geometry()

    @property
    def geometry_transform(self):
        return self._geometry_transform

    @geometry_transform.setter
    def geometry_transform(self, AffineMatrix3D value):
        self._geometry_transform = value
        self._configure_geometry()

    @property
    def integrator(self):
        return self._integrator

    @integrator.setter
    def integrator(self, VolumeIntegrator value):
        self._integrator = value
        self._configure_geometry()

    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, object values):

        # setting the emission models causes ModelManager to notify the Plasma object to configure geometry
        # so no need to explicitly rebuild here
        self._models.set(values)

    @property
    def atomic_data(self):
        return self._atomic_data

    @atomic_data.setter
    def atomic_data(self, AtomicData value):
        self._atomic_data = value
        self._configure_geometry()

    def _configure_geometry(self):

        # detach existing geometry
        # take a copy of self.children as it will be modified when unparenting
        children = self.children.copy()
        for child in children:
            child.parent = None

        # no further work if there are no emission models
        if not list(self._models):
            return

        # check necessary data is available
        if not self._geometry:
            raise ValueError('The plasma must have a defined geometry to be used with an emission model.')

        if not self._atomic_data:
            raise ValueError('The plasma must have an atomic data source to be used with an emission model.')

        # attach geometry to plasma
        self._geometry.parent = self
        self._geometry.name = 'Plasma Geometry'

        # transform geometry if geometry transform present
        if self._geometry_transform:
            self._geometry.transform = self._geometry_transform
            local_to_plasma = self._geometry.to(self)
        else:
            self._geometry.transform = AffineMatrix3D()
            local_to_plasma = None

        # build plasma material
        self._geometry.material = PlasmaMaterial(self, self._atomic_data, list(self._models), self.integrator, local_to_plasma)

    def _modified(self):
        """
        Called when a scene-graph change occurs that modifies this Node's root
        transforms. This will occur if the Node's transform is modified, a
        parent node transform is modified or if the Node's section of scene-
        graph is re-parented.
        """

        # plasma section of the scene-graph has been modified, alert dependents
        self.notifier.notify()