#ifndef _UPDATE_
#define _UPDATE_

#include "globhead.h"

void update(par* p, gsl_vector** dbiases, gsl_matrix** dweights, double step){
  regularisation(p, dbiases, dweights);
  momentum_update(p, dbiases, dweights, step);
}

void ispc_update_bias_helper(par* p, gsl_vector** dbiases, int i, int nrow, double mu, double lambda, double step){
  gsl_vector* res = gsl_vector_alloc(nrow);
  updatebias_ispc((*(p->biases[i])).data,
                  (*(dbiases[i])).data,
                  (*(p->biases_momentum[i])).data,
                  (*res).data,
                  mu, lambda, step, nrow);
  gsl_vector_memcpy(p->biases_momentum[i], res);
  free(res);

  gsl_vector_add(p->biases[i], p->biases_momentum[i]);
}


void ispc_update_W_helper(par* p, gsl_matrix** dW, int i, int nrow, int ncol, double mu, double lambda, double step){
  gsl_matrix* res = gsl_matrix_alloc(nrow, ncol);
  updateweights_ispc((*(p->weights[i])).data,
                     (*(dW[i])).data,
                     (*(p->weights_momentum[i])).data,
                     (*res).data,
                     mu, lambda, step, nrow, ncol);
  gsl_matrix_memcpy(p->weights_momentum[i], res);
  free(res);

  gsl_matrix_add(p->weights[i], p->weights_momentum[i]);
}


void ispc_update(par* p, gsl_vector** dbiases, gsl_matrix** dweights, double step){
  double mu = p->contract_momentum;
  double lambda = p->penalty;

  for (int i = 0; i < p->num_layers-1; i++){
    int nrow = p->layer_sizes[i+1];
    int ncol = p->layer_sizes[i];
    ispc_update_bias_helper(p, dbiases, i, nrow, mu, lambda, step);
    ispc_update_W_helper(p, dweights, i, nrow, ncol, mu, lambda, step);
  }

}

void regularisation(par* p, gsl_vector** biases, gsl_matrix** weights)
{
  for (int i = 0; i < p->num_layers-1; i++){
    //gsl_blas_daxpy(p->penalty, p->biases[i], biases[i]);

    gsl_matrix* temp = gsl_matrix_alloc((*weights[i]).size1,(*weights[i]).size2);
    gsl_matrix_memcpy(temp, p->weights[i]);
    gsl_matrix_scale(temp, p->penalty);
    gsl_matrix_add(weights[i],temp);
    gsl_matrix_free(temp);
  }

}

void momentum_update(par* p, gsl_vector** biases, gsl_matrix** weights, double step)
{
  double mu = p->contract_momentum;
  for (int i = 0; i < p->num_layers-1; i++){
    //gsl_vector_scale(p->biases_momentum[i], mu);
    //gsl_blas_daxpy(step, biases[i], p->biases_momentum[i]);
    gsl_vector_add(p->biases[i], p->biases_momentum[i]);

    gsl_matrix_scale(p->weights_momentum[i], mu);
    gsl_matrix_scale(weights[i],step);
    gsl_matrix_add(p->weights_momentum[i],weights[i]);
    gsl_matrix_add(p->weights[i], p->weights_momentum[i]);
  }
}

#endif
