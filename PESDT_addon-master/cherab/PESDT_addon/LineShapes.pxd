# cython: language_level=3
# raysect imports

from raysect.core.math.function.float cimport Function1D

from cherab.core.model.lineshape cimport LineShapeModel


from scipy.special import hyp2f1
from libc.math cimport sqrt, erf, M_SQRT2, floor, ceil, fabs, expl

cimport cython

cdef double LORENZIAN_CUTOFF_GAMMA = 50.0

cdef class OpaqueGaussianLine(LineShapeModel):
    pass

cdef class OpaqueDeltaLine(LineShapeModel):
    pass

cdef class DeltaLine(LineShapeModel):
    pass

cdef class StarkFunction(Function1D):
    cdef double _a, _x0, _norm

    cdef double STARK_NORM_COEFFICIENT = 4 * LORENZIAN_CUTOFF_GAMMA * hyp2f1(0.4, 1, 1.4, -(2 * LORENZIAN_CUTOFF_GAMMA)**2.5)

cdef class StarkBroadenedLine(LineShapeModel):
    STARK_MODEL_COEFFICIENTS_DEFAULT = {
        (3, 2): (3.71e-18, 0.7665, 0.064),
        (4, 2): (8.425e-18, 0.7803, 0.050),
        (5, 2): (1.31e-15, 0.6796, 0.030),
        (6, 2): (3.954e-16, 0.7149, 0.028),
        (7, 2): (6.258e-16, 0.712, 0.029),
        (8, 2): (7.378e-16, 0.7159, 0.032),
        (9, 2): (8.947e-16, 0.7177, 0.033),
        (4, 3): (1.330e-16, 0.7449, 0.045),
        (5, 3): (6.64e-16, 0.7356, 0.044),
        (6, 3): (2.481e-15, 0.7118, 0.016),
        (7, 3): (3.270e-15, 0.7137, 0.029),
        (8, 3): (4.343e-15, 0.7133, 0.032),
        (9, 3): (5.588e-15, 0.7165, 0.033),


    }
    
    cdef:
        double _aij, _bij, _cij
        double _fwhm_poly_coeff_gauss[7]
        double _fwhm_poly_coeff_lorentz[7]
        double _weight_poly_coeff[6]