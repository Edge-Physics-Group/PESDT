# cython: language_level=3
# raysect imports
from raysect.optical cimport Spectrum, Point3D, Vector3D
from raysect.core.math.function.float cimport Function1D

from cherab.core.model.lineshape cimport LineShapeModel
from .spectrum cimport OpaqueSpectrum


from scipy.special import hyp2f1
from libc.math cimport sqrt, erf, M_SQRT2, floor, ceil, fabs, expl

cimport cython

cdef double LORENZIAN_CUTOFF_GAMMA = 50.0

cpdef OpaqueGaussianLine add_opaque_gaussian_line(double radiance, double absorbance, double Td, double ds, double wavelength, double sigma, OpaqueSpectrum spectrum)

cdef class OpaqueLine(LineShapeModel):
    cdef inline OpaqueSpectrum add_line(self,
                            double radiance,
                            double absorbance,
                            Point3D point,
                            Vector3D direction,
                            OpaqueSpectrum spectrum):
        raise NotImplementedError("The add_line() method has not been implemented.")

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