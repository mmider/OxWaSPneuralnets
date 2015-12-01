#ifndef _AUXILIARY_
#define _AUXILIARY_

#include "globhead.h"

void print_bias(gsl_vector* bias)
{
  int n = (*bias).size;
  Rprintf("\nprinting bias vector:\n");
  for (int i =0; i < n; i++)
    Rprintf("b(%d)=%g\n",i,gsl_vector_get(bias,i));
}

void print_weight(gsl_matrix* weight)
{
  int n = (*weight).size1;
  int k = (*weight).size2;
  Rprintf("\nprinting weight vector:\n");
  for (int i = 0; i < n; i++)
    for (int j =0; j < k; j++)
      Rprintf("m(%d,%d) = %g\n",i,j,gsl_matrix_get(weight,i,j));
}

void sigmoid(gsl_vector* x){
  // takes vector x, computes sigmoid(x) and stores IN PLACE of x
  int len = (*x).size;
  double temp;
  for (int i = 0; i < len; i++){
    temp = gsl_vector_get(x, i);
    gsl_vector_set(x,i, 1/(1+exp(-temp)));
  }
}

void sigmoid_prime(gsl_vector* x, gsl_vector** ans){
  // takes vector x, computes derivative of sigmoid(x) and RETURNS the vector
  int len = (*x).size;
  *ans = gsl_vector_alloc(len);

  double temp;
  for (int i = 0; i < len; i++){
    temp = gsl_vector_get(x,i);
    gsl_vector_set(*ans,i,1/(exp(temp)+2+exp(-temp)));
  }
}

void softmax(gsl_vector* x){
  // takes vector x, computes softmax(x) and stores IN PLACE of x
  int len = (*x).size;
  double max_score = gsl_vector_max(x);
  double temp_probs[len];
  double total = 0;
  for (int i = 0; i < len; i++){
    temp_probs[i] = exp(gsl_vector_get(x,i)-max_score);
    total += temp_probs[i];
  }
  for (int i = 0; i < len; i++){
    gsl_vector_set(x,i,temp_probs[i]/total);
  }
}

void softmax_prime(gsl_vector* probs, int y, gsl_vector* ans){
  gsl_vector_memcpy(ans, probs);
  gsl_vector_set(ans, y, gsl_vector_get(probs,y)-1);
}

double softmax_cost(gsl_vector* probs, int y){
  double loss;
  loss = -log(gsl_vector_get(probs, y));
  return(loss);
}

double squared_error_cost(gsl_vector* probs, int y, gsl_vector* ans){
  double loss = 0.0;
  int n = (*probs).size;
  for (int i = 0; i < n; i++){
    loss += 0.5 * pow(gsl_vector_get(probs,i)-y,2);
  }
  return(loss);
}


void squared_error_prime(gsl_vector* prediction, int y, gsl_vector* ans)
{
  // calulates the derivative of a cost function and RETURNs vector with it

  gsl_vector_memcpy(ans,prediction);
  double temp = gsl_vector_get(ans, y);
  gsl_vector_set(ans, y, temp - 1);
}


void data_to_gsl_vectors(double* input_array, int nrow, int ncol, gsl_vector* out[]){
	// Fill the matrix element-by-element
	for (int i=0; i<nrow; i++){
		out[i] = gsl_vector_alloc(ncol);
		for (int j=0; j<ncol; j++){
			gsl_vector_set(out[i], j, input_array[j*nrow + i]);
		}
	}
}


gsl_vector* array_to_gsl_vector(double* input_array, int n){
	// Create an empty matrix
	gsl_vector* out = gsl_vector_alloc(n);
	// Fill the matrix element-by-element
	for (int j=0; j<n; j++){
		gsl_vector_set(out, j, input_array[j]);
	}
	return(out);
}


void vec_to_mat(gsl_vector* vec, gsl_matrix** ans)
{
  int n = (*vec).size;
  *ans = gsl_matrix_alloc(1,n);
  for (int i = 0; i < n; i ++)
    gsl_matrix_set(*ans,0,i,gsl_vector_get(vec,i));
}


double correct_guesses(gsl_vector* test_data[],
		       gsl_vector* ys, gsl_vector* biases[],
		       gsl_matrix* weights[], int nrow, int num_layers,
		       int * layer_sizes)
{
  par p;
  par_c q;
  p.weights = weights;
  p.biases = biases;
  p.num_layers = num_layers;
  p.layer_sizes = layer_sizes;
  p.trans = sigmoid;
  p.trans_prime = sigmoid_prime;
  p.trans_final = softmax;
  p.trans_final_prime = sigmoid_prime;
  p.cost = softmax_cost;
  p.cost_prime = softmax_prime;
  p.total_cost = 0;
  q.total_cost = 0;
  gsl_vector* z[num_layers];
  gsl_vector* transf_x[num_layers];

  init_bias_object(z, layer_sizes, num_layers);
  init_bias_object(transf_x, layer_sizes, num_layers);

  q.z = z;
  q.transf_x = transf_x;


  double total = 0.0;
  for (int i = 0; i < nrow; i++){
    q.x = test_data[i];
    q.y = (int) gsl_vector_get(ys,i);
    forward(&q, &p);

    int y_fitted = gsl_vector_max_index(q.transf_x[num_layers-1]);
    total += (q.y==y_fitted);
  }

  for (int i = 0; i < num_layers - 1; i++){
    gsl_vector_free(z[i]);
    gsl_vector_free(transf_x[i]);
  }
  return total / nrow;
}


#endif // _AUXILIARY_
