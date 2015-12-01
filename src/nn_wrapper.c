#ifndef _NN_WRAPPER_
#define _NN_WRAPPER_

#include "globhead.h"

void nn(double* train_data, double* ys, double* test_data, double* ys_test, int* layer_sizes,
	int* num_layers, int* num_iterations, int* core_num, double* step_size,
	int* nrow, int* ncol, int* nrow_test, double* penalty, double* output,
	double* output2, double* output3, int* trans_type)
{
   gsl_vector* data_vectors[*nrow];
   data_to_gsl_vectors(train_data, *nrow, *ncol, data_vectors);

   gsl_vector* y = array_to_gsl_vector(ys, *nrow);

   gsl_matrix* output_weights[*num_layers-1];
   gsl_vector* output_biases[*num_layers-1];

   init_bias_object(output_biases, (layer_sizes+1), *num_layers-1);
   init_weight_object(output_weights, layer_sizes, *num_layers);

   NeuralNets(layer_sizes,*num_layers,data_vectors,y, *num_iterations, *core_num,
	      *step_size, output_weights, output_biases,*nrow,*ncol,*penalty, output2, *trans_type);
   *output = correct_guesses(data_vectors, y, output_biases, output_weights, *nrow, *num_layers, layer_sizes, *trans_type);
   gsl_vector* data_test_vectors[*nrow_test];
   data_to_gsl_vectors(test_data, *nrow_test, *ncol, data_test_vectors);
   gsl_vector* y_test = array_to_gsl_vector(ys_test, *nrow_test);
   *output3 = correct_guesses(data_test_vectors, y_test, output_biases, output_weights, *nrow_test, *num_layers, layer_sizes, *trans_type);

   // destroy biases and weights
}

#endif // _NN_WRAPPER_
