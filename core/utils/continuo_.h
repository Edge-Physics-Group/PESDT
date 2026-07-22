#ifndef CONTINUO_H
#define CONTINUO_H
#include <vector>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct ContinuumRadiation
{
    double free_free;
    double free_bound;
} ContinuumRadiation;
ContinuumRadiation continuo_(double wavelength_A,double Te_eV,int atomic_number,int ion_charge);

void continuov_(double *wavelength_A,double *Te_eV,int atomic_number,int ion_charge, size_t num_wl, size_t num_te, ContinuumRadiation* output);

void continuovpar_(double *wavelength_A,double *Te_eV,int atomic_number,int ion_charge, size_t num_wl, size_t num_te, ContinuumRadiation* output);

#ifdef __cplusplus
}
#endif

#endif