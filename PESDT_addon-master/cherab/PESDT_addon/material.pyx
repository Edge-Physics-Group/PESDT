

from raysect.optical cimport World, Primitive, Ray, Spectrum, SpectralFunction, Point3D, Vector3D, AffineMatrix3D
from .inhomogenous cimport InhomogeneousVolumeEmitter
from raysect.optical.material.emitter.inhomogeneous cimport VolumeIntegrator
from cherab.core.plasma.model cimport PlasmaModel
from .node cimport OpaquePlasma
from cherab.core.atomic cimport AtomicData
from .PlasmaModel cimport OpaquePlasmaModel
from .spectrum cimport OpaqueSpectrum

cdef class OpaquePlasmaMaterial(InhomogeneousVolumeEmitter):
    """Raysect Material that handles the integration of the plasma model emission."""

    def __init__(self, OpaquePlasma plasma not None, AtomicData atomic_data not None, list models not None, VolumeIntegrator integrator not None, AffineMatrix3D local_to_plasma):

        super().__init__(integrator)

        self._plasma = plasma
        self._atomic_data = atomic_data
        self._local_to_plasma = local_to_plasma

        # validate
        for model in models:
            if not isinstance(model, OpaquePlasmaModel):
                raise TypeError('Model supplied to PlasmaMaterial is not a PlasmaModel.')

        # configure models
        for model in models:
            model.plasma = plasma
            model.atomic_data = atomic_data

        self._models = models

    cpdef OpaqueSpectrum emission_function(self, Point3D point, Vector3D direction, OpaqueSpectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix3D to_local, AffineMatrix3D to_world):

        cdef PlasmaModel model

        # perform coordinate transform to plasma space if required
        if self._local_to_plasma:
            point = point.transform(self._local_to_plasma)
            direction = direction.transform(self._local_to_plasma)

        # call each model and accumulate spectrum
        for model in self._models:
            spectrum = model.emission(point, direction, spectrum)

        return spectrum