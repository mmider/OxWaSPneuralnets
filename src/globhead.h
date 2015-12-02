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
#include <omp.h>
#include "timing.h"

typedef struct parameters
{
  gsl_matrix** weights;
  gsl_vector** biases;
  gsl_matrix** weights_momentum;
  gsl_vector** biases_momentum;

  double step_size;
  double penalty;
  double total_cost;
  double contract_momentum;

  int* layer_sizes;
  int num_layers;
  int batch_number; // number of cores
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

void destroy_parameters(par* p);

void destroy_parameters_core(par_c* q, par* p);

void NeuralNets(int* layer_sizes, int num_layers, gsl_vector* train_data[],
		gsl_vector* ys,int num_iterations, int core_num,
		double step_size,gsl_matrix* output_weights[], gsl_vector* output_biases[],
		int nrow, int ncol, double penalty, double cost_hist[], int transformation_type);

void forward(par_c* q, par* p);

void compute_score(par_c* q, par* p, int i);

void backpropagation (par_c* q, par* p);

void update_last_delta(par_c* q, par* p);

void update_gradients(par_c* q, par*p);

void BatchUpdate(gsl_vector* train_data[], gsl_vector* ys, par* p);

void regularisation(par* p, gsl_vector** biases, gsl_matrix** weights);

void momentum_update(par* p, gsl_vector** biases, gsl_matrix** weights, double step);

void squared_error_prime(gsl_vector* prediction, int y, gsl_vector* ans);

double squared_error_cost(gsl_vector* probs, int y);

double cost_regul_term(par* p);

void vec_to_mat(gsl_vector* vec, gsl_matrix** ans);

double evaluate_results(gsl_vector* test_data[],
		       gsl_vector* ys, gsl_vector* biases[],
			gsl_matrix* weights[], int nrow, int ncat, int num_layers,
			int * layer_sizes,  int transformation_type, double probs[],
			int predicted[]);

void data_to_gsl_vectors(double* input_array, int nrow, int ncol, gsl_vector* out[]);

void nn(double* train_data, double* ys, double* test_data, double* ys_test,
	int* layer_sizes, int* num_layers, int* num_iterations,
	int* core_num, double* step_size, int* nrow, int* ncol,
	int* nrow_test, double* penalty, double* output,
	double* output2, double* output3,double* output4, double* output5,
	int* output6, int* output7, int* trans_type);

gsl_vector* array_to_gsl_vector(double* input_array, int n);

int* randomperm11(int n);

int* rand_fold(const int N_obs, const int fold);

void SplitFoldfunc(const gsl_matrix *TrainData, int fold, int* rand_seq, gsl_matrix** SubTrain);

void combinefold (gsl_matrix** foldX, gsl_matrix** foldY, int N_obs, int fold, int SizeGroup, int Features,
                  int label_size, gsl_matrix** CvTrainX, gsl_matrix** CvTrainY);

void CrossVal(const gsl_matrix* XTrainData, const gsl_matrix* YTrainData, const gsl_matrix* XTestData,
              const gsl_matrix* YTestData, const int FOLD, const double* Lambda, const int sizelambda, const int* layer_sizes,  const int num_layers,
              const int num_iterations, const int core_size, const double step_size, const int trans_type, double* probs_test, int* predicted_test);

void CvNN(double* train_data, double* ys, int* layer_sizes, int* num_layers, int* num_iterations,
            int* core_num, double* step_size, int* nrow, int* ncol, double* penalty, int* penalty_size,
            int* trans_type, double* test_data_x, double* test_data_y, int* nrow_test, int* fold, double* probs_test, int* predicted_test);

#endif // _GLOBHEAD_H_
