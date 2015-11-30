#ifndef _NEURALNETS_
#define _NEURALNETS_

#include "globhead.h"

void NeuralNets(int* layer_sizes, int num_layers, gsl_vector* train_data[],
		gsl_vector* ys, int num_iterations, int batch_size,
		double step_size, gsl_matrix* output_weights[], gsl_vector* output_biases[],
		int nrow, int ncol, double penalty, double cost_hist[]
		//int transformation_function, int final_transformation
		)
{
  /*
    Takes: layer_sizes -    each element of a vector specifices the number of units in a layer,
                            the first layer must be of size equal to dimensionality of x, the
			    last must be of number of possible categories y may take.
	   num_layers -     number of layers, additional layer for the original x-es must be
	                    counted (and also included in layer_sizes as specified above)
	   train_data -     array of pointers, which point to separate x observations
	   ys -             elements of vector give labels to corresponding observations
	   num_iterations - number of times the full sweep of Stochastic Gradient Descent
	                    should run when fitting the model
	   batch_size -     number of observations in a batch - each core will receive one batch
	   step_size -      currently fixed parameter for Stoch Grad Desc,
	                    TO DO will be changed in later versions
	   output_weigts -  fitted weights will go here
	   output_biases -  fitted biases will go here
	   nrow -           number of x observations
	   ncol -           dimensionality of x

    Fits neural network using backpropagation and stochastic gradient descent
    algorithms for arbitrary number of layers and units in layers. Currently supports
    only classification problems
  */

  // all auxiliary parameters are stored here:
  par p;
  // set up the random number generator
  const gsl_rng_type* T;
  gsl_rng* r;
  gsl_rng_env_setup();
  T = gsl_rng_default;
  r = gsl_rng_alloc(T);
  gsl_rng_set(r, time(NULL));

  // initialize a list of vectors with biases
  // and a list of matrices with weights
  // there are no biases in the first layer so start from layer_size + 1
  gsl_vector* biases[num_layers-1];
  gsl_matrix* weights[num_layers-1];
  init_bias_object(biases, (layer_sizes+1),num_layers-1);
  init_weight_object(weights, layer_sizes,num_layers);

  // set initial values
  randomize_bias(biases, (layer_sizes+1), num_layers-1, r);
  randomize_weight(weights, layer_sizes, num_layers-1, r);

  // set parameters that will be passed around:
  p.biases = biases;
  p.weights = weights;
  p.layer_sizes = layer_sizes;
  p.num_layers = num_layers;
  p.batch_size = batch_size;
  p.step_size = step_size;
  p.r = r;
  p.trans = sigmoid;
  p.trans_prime = sigmoid_prime;
  p.trans_final = softmax;
  p.trans_final_prime = sigmoid_prime;
  p.penalty = penalty;
  p.cost = softmax_cost;
  /*
  if (transformation_function == 0){
    p.trans = relu;
    p.trans_prime = relu_prime;
  }
  else{
    p.trans = sigmoid;
    p.trans_prime = sigmoid.prime;
  }
  if (final_transformation == 0){
    p.trans_final = softmax;
    p.trans_final_prime = softmax_prime;
  }
  else {
    p.trans_final = sigmoid;
    p.trans_final_prime = sigmoid.prime;
  }
  */
  p.cost_prime = softmax_prime;
  p.nrow = nrow;
  p.ncol = ncol;
  // perform Stochasstic Gradient Descents num_iterations many times
  
  for (int i = 0; i < num_iterations; i++){
    p.total_cost = 0.0;
    StochGradDesc(train_data, ys, &p);
    cost_hist[i] = p.total_cost;
    //Rprintf("%g\n",p.total_cost);
  }

  // copy the results to the output matrices and vectors
  for (int i = 0; i < num_layers-1; i++){
    gsl_vector_memcpy(output_biases[i], p.biases[i]);
    gsl_matrix_memcpy(output_weights[i], p.weights[i]);
  }
} 


void StochGradDesc(gsl_vector* train_data[], gsl_vector* ys, par* p){
  /*
    Takes the data: train_data is an array of pointers to vectors containing
                    separate x variables,
		    ys is a vector where in each row gives the label for a
		    corresponding x variable,
    and additional parameters stored in p,
    nrow - number of x observations
    ncol - dimension of each x observation
    
    performs one full sweep of Stochastic Gradient Descent:
    shuffles the data, splits into batches, (TO DO: sends each batch to separate core)
    performs forward propagation and backward propagation for each observation within
    batch, updates biases and weights vectors and repeats for as many batches as can
    be fit in the nrow without any repeats
  */

  // shuffle the data:
  // the data will be drawn in order specified by the consecutive
  // numbers from the indices vector
  int m = (*p).nrow;
  int indices[m];
  for (int i = 0; i < m; i++){
    indices[i] = i;
  }
  gsl_ran_shuffle((*p).r,indices, m, sizeof(int));

  int batch_number = m / (*p).batch_size;
  
  for (int i = 0; i < batch_number; i++){
    // SGD update for a batch, this will be done by each core separately:
    // ----------------------------
    
    // initialise parameters that will be shared in each core
    par_c q;
    q.total_cost = 0;
    // init_parameters_core(&q, p);
    int t = (*p).num_layers;
    int* ls = (*p).layer_sizes;
    gsl_vector* gradient_biases[t-1];
    gsl_matrix* gradient_weights[t-1];
    gsl_vector* z[t];
    gsl_vector* transf_x[t];
    gsl_vector* delta[t];
  
    init_bias_object(gradient_biases, ls+1,t-1);
    init_weight_object(gradient_weights, ls,t);
    init_bias_object(z, ls, t);
    init_bias_object(transf_x, ls, t);
    init_bias_object(delta, ls+1, t - 1);

    q.gradient_biases = gradient_biases;
    q.gradient_weights = gradient_weights;
    q.z = z;
    q.transf_x = transf_x;
    q.delta = delta;
    
    // initialise objects that will be passed around:
    // gradients, z - which stores z=Wx+b from each layer, and transf_x=sigmoid(z)
    // for each observation from the batch perform forward and backward sweep
    for (int j = 0; j < (*p).batch_size; j++){
      q.x = train_data[i * (*p).batch_size + j];
      q.y = (int) gsl_vector_get(ys,i * (*p).batch_size + j);
      forward(&q,p);
      backpropagation(&q,p);
    }
    (*p).total_cost += q.total_cost;

    // TO DO
    // at this point the cores should communicate to update weights and biases
    // so the update shuould really happen outside the loop
    // update weights and biases:
    for (int j = 0; j < (*p).num_layers-1; j++){
      q.learning_rate = -(*p).step_size/ (*p).batch_size;
      gsl_blas_daxpy(q.learning_rate, q.gradient_biases[j], (*p).biases[j]);
      gsl_matrix_scale(q.gradient_weights[j], q.learning_rate);
      // apply penalty
      gsl_matrix_scale((*p).weights[j],1-(*p).penalty * q.learning_rate);
      gsl_matrix_add((*p).weights[j], q.gradient_weights[j]);
    }
    destroy_parameters_core(&q,p);
  }
}

#endif //_NEURALNETS_
