#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include <gsl/gsl_sf_erf.h>
#include <gsl/gsl_math.h>

#define LOG2PI (M_LN2 + M_LNPI)
#define LOGPI_2 (M_LNPI - M_LN2)

double log_marg_amp_full_C(int Nmodels, double modelModel[], double dataModel[], double sigma, unsigned int lastHalfRange);

double log_marg_amp_except_final_C(int Nmodels, double modelModel[], double dataModel[], double sigma);