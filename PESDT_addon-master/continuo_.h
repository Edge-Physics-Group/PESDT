#ifndef CONTINUO_H
#define CONTINUO_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ContinuumRadiation
{
    double free_free;
    double free_bound;
} ContinuumRadiation;

ContinuumRadiation continuo_(double wavelength_A,double Te_eV,int atomic_number,int ion_charge);


#ifdef __cplusplus
}
#endif

#endif