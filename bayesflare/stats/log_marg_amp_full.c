#include "log_marg_amp_full.h"

double log_marg_amp_full_C(int Nmodels, double modelModel[], double dataModel[], double sigma, unsigned int lastHalfRange){
  /* coefficients of squares of model amplitudes */
  double squared[Nmodels];

  /* coefficients of model amplitudes */
  double coeffs[Nmodels][Nmodels];
  memset(coeffs, 0, sizeof(coeffs[0][0]) * Nmodels * Nmodels); /* initialise all values to zero */

  int i = 0, j = 0, k = 0, nm = Nmodels-1;

  double X = 0., invX = 0., invTwoX = 0., invFourX = 0., Y = 0., Z = 0., logL = 0.;

  /* set up coeffs matrix */
  for ( i=0; i<Nmodels; i++ ){
    squared[i] = modelModel[i*Nmodels + i];
    coeffs[i][i] = -2.*dataModel[i];

    if( !isfinite(squared[i]) || !isfinite(coeffs[i][i]) ){ return -INFINITY; }

    for ( j=(i+1); j<Nmodels; j++ ){
      coeffs[i][j] = 2.*modelModel[i*Nmodels + j];

      if( !isfinite(coeffs[i][j]) ){ return -INFINITY; }
    }
  }

  for ( i=0; i<nm; i++ ){
    X = squared[i];
    invX = 1./X;
    invTwoX = 0.5*invX;
    invFourX = 0.25*invX;

    /* get the coefficients from the Y^2 term */
    for ( j=i; j<Nmodels; j++ ){
      for ( k=i; k<Nmodels; k++ ){
        /* add on new coefficients of squared terms */
        if ( j == i ){
          if ( k > j ){ squared[k] -= coeffs[j][k]*coeffs[j][k]*invFourX; }
          else if ( k == j ){ Z -= coeffs[j][k]*coeffs[j][k]*invFourX; }
        }
        else {
          if ( k == i ){ coeffs[j][j] -= coeffs[i][j]*coeffs[i][k]*invTwoX; }
          else if ( k > j ){ coeffs[j][k] -= coeffs[i][j]*coeffs[i][k]*invTwoX; }
        }
      }
    }
  }

  X = squared[nm];
  Y = coeffs[nm][nm];

  /* calculate analytic integral and get log likelihood */
  for ( i=0; i<Nmodels; i++ ){ logL -= 0.5*log(squared[i]); }

  logL -= 0.5*(Z - 0.25*Y*Y/X) / (sigma*sigma);

  logL += Nmodels*log(sigma);

  /* check whether final model is between 0 and infinity or -infinity and infinity */
  if (lastHalfRange == 1 ){
    logL += 0.5*(double)nm*LOG2PI + 0.5*LOGPI_2 + gsl_sf_log_erfc(0.5*Y/(sigma*sqrt(2.*X)));
  }
  else{ logL += 0.5*(double)Nmodels*LOG2PI; }

  return logL;
}


double log_marg_amp_except_final_C(int Nmodels, double modelModel[], double dataModel[], double sigma){
  /* coefficients of squares of model amplitudes */
  double squared[Nmodels];

  /* coefficients of model amplitudes */
  double coeffs[Nmodels][Nmodels];
  memset(coeffs, 0, sizeof(coeffs[0][0]) * Nmodels * Nmodels); /* initialise all values to zero */

  int i = 0, j = 0, k = 0, nm = Nmodels-1, nmm = Nmodels-2;

  double X = 0., invX = 0., invTwoX = 0., invFourX = 0., Y = 0., Z = 0., logL = 0.;

  /* set up coeffs matrix */
  for ( i=0; i<Nmodels; i++ ){
    squared[i] = modelModel[i*Nmodels + i];
    coeffs[i][i] = -2.*dataModel[i];

    if( !isfinite(squared[i]) || !isfinite(coeffs[i][i]) ){ return -INFINITY; }

    for ( j=(i+1); j<Nmodels; j++ ){
      coeffs[i][j] = 2.*modelModel[i*Nmodels + j];

      if( !isfinite(coeffs[i][j]) ){ return -INFINITY; }
    }
  }

  for ( i=0; i<nmm; i++ ){
    X = squared[i];
    invX = 1./X;
    invTwoX = 0.5*invX;
    invFourX = 0.25*invX;

    /* get the coefficients from the Y^2 term */
    for ( j=i; j<Nmodels; j++ ){
      for ( k=i; k<Nmodels; k++ ){
        /* add on new coefficients of squared terms */
        if ( ( j != i+1 && j != nmm ) && ( k != i+1 && k != nmm ) ){ Z -= coeffs[i][j]*coeffs[i][k]*invFourX; }

        if ( j == i ){
          if ( k > j && k < nm ){ squared[k] -= coeffs[j][k]*coeffs[j][k]*invFourX; }
        }
        else {
          if ( k == i && j < nm ){ coeffs[j][j] -= coeffs[i][j]*coeffs[i][k]*invTwoX; }
          else if ( k > j ){ coeffs[j][k] -= coeffs[i][j]*coeffs[i][k]*invTwoX; }
        }
      }
    }
  }

  X = squared[nmm];
  for ( i=nmm; i<Nmodels; i++ ){ Y += coeffs[nmm][i]; }
  Z += (squared[nm] + coeffs[nm][nm]);

  /* calculate analytic integral and get log likelihood */
  for ( i=0; i<nm; i++ ){ logL -= 0.5*log(squared[i]); }

  logL -= 0.5*(Z - 0.25*Y*Y/X) / (sigma*sigma);

  logL += nm*log(sigma);

  logL += 0.5*(double)nm*LOG2PI;

  return logL;
}
