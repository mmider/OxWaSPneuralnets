#ifndef _INIT_
#define _INIT_

#include "globhead.h"

void init_bias_object(gsl_vector* vec[], int* layer_sizes, int n)
{
  // allocate memory for each layer in layer_sizes to store the biases of units
  // n is the number of layers-1
  for (int i = 0; i < n; i++){
    vec[i] = gsl_vector_calloc(layer_sizes[i]);
  }
}

void init_weight_object(gsl_matrix* mat[], int* layer_sizes,int n)
{
  // allocate memory for each layer in layer_sizes to store the weights of units
  // n is the number of layers
  for (int i = 0; i < n-1; i++){
    mat[i] = gsl_matrix_calloc(layer_sizes[i+1],layer_sizes[i]);
  }
}

void randomize_bias(gsl_vector* biases[],
		    int* layer_sizes,
		    int n,
		    gsl_rng* r)
{
  // set initial values of biases to N(0,0.1)
  for (int i = 0; i < n; i++)
    for (int j = 0; j < layer_sizes[i]; j++){
      gsl_vector_set(biases[i], j, gsl_ran_gaussian(r, 0.1));
    }
}

void randomize_weight(gsl_matrix* weights[],
		      int* layer_sizes,
		      int n,
		      gsl_rng* r)
{
  // set initial values of weights to N(0,0.1)
  for (int i = 0; i < n; i++)
    for (int j = 0; j < layer_sizes[i+1]; j++)
      for (int k = 0; k < layer_sizes[i]; k++)
	gsl_matrix_set(weights[i],j,k,gsl_ran_gaussian(r,0.1));
}

void destroy_bias_obj(gsl_vector* biases[], int n)
{
  // free memory
  for (int i = 0; i < n; i++)
    gsl_vector_free(biases[i]);
}

void destroy_weight_obj(gsl_matrix* weights[], int n)
{
  // free memory
  for (int i = 0; i < n; i++)
    gsl_matrix_free(weights[i]);
}

void destroy_parameters_core(par_c* q, par* p)
{
  int n = p->num_layers;
  destroy_bias_obj(q->gradient_biases,n - 1);
  destroy_weight_obj(q->gradient_weights,n - 1);
  destroy_bias_obj(q->z,n);
  destroy_bias_obj(q->transf_x, n);
  destroy_bias_obj(q->delta, n-1);
}


#endif // _INIT_
