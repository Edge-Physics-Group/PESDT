# cython: language_level=3
# raysect imports
from raysect.optical cimport Spectrum, Point3D, Vector3D
from raysect.core.math.function.float cimport Function1D
from cherab.core.math.integrators cimport Integrator1D
from cherab.core.model.lineshape cimport LineShapeModel
from cherab.core.atomic cimport Line
from cherab.core.species cimport Species
from cherab.core.plasma cimport Plasma
from cherab.core.atomic cimport AtomicData

from scipy.special import hyp2f1
from libc.math cimport sqrt, erf, M_SQRT2, floor, ceil, fabs, expl

cimport cython

cdef class DeltaLine(LineShapeModel):
    pass

cdef class StarkFunction(Function1D):
    cdef double STARK_NORM_COEFFICIENT
    cdef double _a, _x0, _norm

cdef class StarkBroadenedLine(LineShapeModel):
    
    
    cdef:
        double _aij, _bij, _cij
        double _fwhm_poly_coeff_gauss[7]
        double _fwhm_poly_coeff_lorentz[7]
        double _weight_poly_coeff[6]