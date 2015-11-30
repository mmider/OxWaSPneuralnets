#ifndef _BACKPROPAGATION_
#define _BACKPROPAGATION_

#include "globhead.h"


void backpropagation (par_c* q, par* p)
{
  /*
    Takes x observation and a corresponding label y, as well as z and transf_x from
    the most recent forward pass. Calculates deltas for each unit and updates
    the cumulative gradient of biases and weights of a given batch
  */
  // Output layer first
  gsl_vector* cp;
  gsl_vector* sp;
  if (false){
    cp = gsl_vector_alloc((*p).layer_sizes[(*p).num_layers-1]);
    (*p).cost_prime((*q).transf_x[(*p).num_layers-1],(*q).y, cp);
    (*p).trans_final_prime((*q).z[(*p).num_layers-1], &sp);
    gsl_vector_mul(cp,sp);
    gsl_vector_memcpy((*q).delta[(*p).num_layers - 2], cp);
    gsl_vector_free(cp);
    gsl_vector_free(sp);
  }
  else {
    (*p).cost_prime((*q).transf_x[(*p).num_layers-1],(*q).y,(*q).delta[(*p).num_layers-2]);
  }
  // For previous layers
  for (int l = (*p).num_layers - 2; l > 0; l--){
    (*p).trans_prime((*q).z[l], &sp);
    gsl_blas_dgemv(CblasTrans,1,(*p).weights[l],(*q).delta[l],0,(*q).delta[l-1]);
    gsl_vector_mul((*q).delta[l-1],sp);
    gsl_vector_free(sp);
  }
  // Computation of the error function derivatives
  gsl_matrix* delta_temp;
  gsl_matrix* a_temp;

  for (int l = 0; l < (*p).num_layers - 1; l++){
    gsl_vector_add((*q).gradient_biases[l],(*q).delta[l]);
    delta_temp = gsl_matrix_alloc(1,(*p).layer_sizes[l+1]);
    vec_to_mat((*q).delta[l], &delta_temp);
    vec_to_mat((*q).transf_x[l], &a_temp);

    gsl_blas_dgemm(CblasTrans,CblasNoTrans,1,delta_temp, a_temp,1,(*q).gradient_weights[l]);
    gsl_matrix_free(delta_temp);
    gsl_matrix_free(a_temp);
  }
}

#endif // _BACKPROPAGATION_
