# cython: language_level=3
# raysect imports
from raysect.optical cimport Spectrum, Point3D, Vector3D
from raysect.core.math.function.float cimport Function1D
from cherab.core.math.integrators cimport Integrator1D
from cherab.core.model.lineshape cimport LineShapeModel
from .spectrum cimport OpaqueSpectrum
from cherab.core.atomic cimport Line
from cherab.core.species cimport Species
from cherab.core.plasma cimport Plasma
from cherab.core.atomic cimport AtomicData

from scipy.special import hyp2f1
from libc.math cimport sqrt, erf, M_SQRT2, floor, ceil, fabs, expl

cimport cython

cdef double LORENZIAN_CUTOFF_GAMMA = 50.0

cpdef OpaqueGaussianLine add_opaque_gaussian_line(double radiance, double absorbance, double Td, double ds, double wavelength, double sigma, OpaqueSpectrum spectrum)

cdef class OpaqueLine:

    cdef:
        Line line
        double wavelength
        Species target_species
        Plasma plasma
        AtomicData atomic_data
        Integrator1D integrator

    cpdef OpaqueSpectrum add_line(self,
                            double radiance,
                            double absorbance,
                            Point3D point,
                            Vector3D direction,
                            OpaqueSpectrum spectrum)


cdef class OpaqueGaussianLine(OpaqueLine):
    pass

cdef class OpaqueDeltaLine(OpaqueLine):
    pass

cdef class DeltaLine(LineShapeModel):
    pass

cdef class StarkFunction(Function1D):
    cdef double _a, _x0, _norm

    cdef double STARK_NORM_COEFFICIENT

cdef class StarkBroadenedLine(LineShapeModel):
    
    
    cdef:
        double _aij, _bij, _cij
        double _fwhm_poly_coeff_gauss[7]
        double _fwhm_poly_coeff_lorentz[7]
        double _weight_poly_coeff[6]