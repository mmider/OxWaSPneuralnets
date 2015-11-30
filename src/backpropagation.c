#ifndef _BACKPROPAGATION_
#define _BACKPROPAGATION_

#include "globhead.h"


void backpropagation (par_c* q, par* p)
{
  /*
    for observation (x,y) updates the cumulative
    gradient of biases and weights. Calculates
    deltas along the way.
  */

  // Output layer first
  update_last_delta(q,p);
  
  // For previous layers
  gsl_vector* cp;
  gsl_vector* sp;
  for (int l = p->num_layers - 2; l > 0; l--){
    p->trans_prime(q->z[l], &sp);
    // delta(l-1) = (W(l)'.delta(l)) * sigmoid'(z(l))
    gsl_blas_dgemv(CblasTrans,1,p->weights[l],q->delta[l],0,q->delta[l-1]);
    gsl_vector_mul(q->delta[l-1],sp);
    gsl_vector_free(sp);
  }

  update_gradients(q,p);
}

void update_last_delta(par_c* q, par* p)
{
  int n = p->num_layers;
  if (p->transformation_type == 1){
    // cross entropy loss and softmax transformation
    p->cost_prime(q->transf_x[n-1],q->y,q->delta[n-2]);
  }
  else{
    // squared error loss and sigmoid transformation
    gsl_vector* cp;
    gsl_vector* sp;
    cp = gsl_vector_alloc(p->layer_sizes[n-1]);
    // derivative of squared error loss
    p->cost_prime(q->transf_x[n-1],q->y, cp);
    // derivative of sigmoid
    p->trans_final_prime(q->z[n-1], &sp);
    // delta
    gsl_vector_mul(cp,sp);
    gsl_vector_memcpy(q->delta[n - 2], cp);
    gsl_vector_free(cp);
    gsl_vector_free(sp);
  }
}


void update_gradients(par_c* q, par*p)
{
  gsl_matrix* delta_temp;
  gsl_matrix* a_temp;

  for (int l = 0; l < p->num_layers - 1; l++){
    gsl_vector_add(q->gradient_biases[l],q->delta[l]);
    delta_temp = gsl_matrix_alloc(1,p->layer_sizes[l+1]);
    vec_to_mat(q->delta[l], &delta_temp);
    vec_to_mat(q->transf_x[l], &a_temp);
    gsl_blas_dgemm(CblasTrans,CblasNoTrans,1,delta_temp, a_temp,1,q->gradient_weights[l]);
    gsl_matrix_free(delta_temp);
    gsl_matrix_free(a_temp);
  }  
}



#endif // _BACKPROPAGATION_
