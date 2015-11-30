#ifndef _GLOBHEAD_H_
#define _GLOBHEAD_H_

#include <stdio.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_blas.h>
#include <math.h>
#include <stdbool.h>
#include <R.h>
#include <time.h>

typedef struct parameters
{
  gsl_matrix** weights;
  gsl_vector** biases;
  double step_size;
  double penalty;
  double total_cost;
  
  int* layer_sizes;
  int num_layers;
  int batch_size;
  int nrow;
  int ncol;
  int transformation_type;
  
  gsl_rng* r;

  void (*trans)(gsl_vector* x);
  void (*trans_final)(gsl_vector* x);

  void (*trans_prime)(gsl_vector* x, gsl_vector** ans);
  void (*trans_final_prime)(gsl_vector* x, gsl_vector** ans);

  double (*cost)(gsl_vector* x, int y);
  void (*cost_prime)(gsl_vector* x, int y, gsl_vector* ans);

}par;

typedef struct parameters_core
{
  gsl_vector** gradient_biases;
  gsl_matrix** gradient_weights;
  gsl_vector** z;
  gsl_vector** transf_x;
  gsl_vector** delta;
  double total_cost;

  double learning_rate;
  gsl_vector* x;
  int y;
  
} par_c;

void print_bias(gsl_vector* bias);

void print_weight(gsl_matrix* weight);

void sigmoid(gsl_vector* x);

void sigmoid_prime(gsl_vector* x, gsl_vector** ans);

void softmax(gsl_vector* x);

void softmax_prime(gsl_vector* probs, int y, gsl_vector* ans);

double softmax_cost(gsl_vector* probs, int y);

void init_bias_object(gsl_vector* vec[], int* layer_sizes, int n);

void init_weight_object(gsl_matrix* mat[], int* layer_sizes, int n);

void randomize_bias(gsl_vector* biases[],
		    int* layer_sizes,
		    int num_lay,
		    gsl_rng* r);

void randomize_weight(gsl_matrix* weights[],
		      int* layer_sizes,
		      int n,
		      gsl_rng* r);

void destroy_bias_obj(gsl_vector* biases[], int n);

void destroy_weight_obj(gsl_matrix* weights[], int n);


void destroy_parameters_core(par_c* q, par* p);

void NeuralNets(int* layer_sizes, int num_layers, gsl_vector* train_data[],
		gsl_vector* ys,int num_iterations, int batch_size,
		double step_size,gsl_matrix* output_weights[], gsl_vector* output_biases[],
		int nrow, int ncol, double penalty, double cost_hist[], int transformation_type);

void forward(par_c* q, par* p);

void compute_score(par_c* q, par* p, int i);

void backpropagation (par_c* q, par* p);

void update_last_delta(par_c* q, par* p);

void update_gradients(par_c* q, par*p);

void StochGradDesc(gsl_vector* train_data[], gsl_vector* ys, par* p);

void squared_error_prime(gsl_vector* prediction, int y, gsl_vector* ans);

double squared_error_cost(gsl_vector* probs, int y, gsl_vector* ans);

void vec_to_mat(gsl_vector* vec, gsl_matrix** ans);

double correct_guesses(gsl_vector* test_data[],
		       gsl_vector* ys, gsl_vector* biases[],
		       gsl_matrix* weights[], int nrow, int num_layers,
					      int * layer_sizes);

void data_to_gsl_vectors(double* input_array, int nrow, int ncol, gsl_vector* out[]);

void nn(double* train_data, double* ys, int* layer_sizes, int* num_layers, int* num_iterations,
	int* batch_size, double* step_size, int* nrow, int* ncol, double* penalty, double* output, double* output2, int* trans_type);

gsl_vector* array_to_gsl_vector(double* input_array, int n);

#endif // _GLOBHEAD_H_
