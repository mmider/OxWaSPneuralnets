#ifndef _NEURALNETS_
#define _NEURALNETS_

#include "globhead.h"

void NeuralNets(int* layer_sizes, int num_layers, gsl_vector* train_data[],
		gsl_vector* ys, int num_iterations, int core_num,
		double step_size, gsl_matrix* output_weights[], gsl_vector* output_biases[],
		int nrow, int ncol, double penalty, double cost_hist[], int transformation_type)
{
  /*
    Takes: layer_sizes -    each element of a vector specifices the number of units in a layer,
                            the first layer must be of size equal to dimensionality of x, the
			    last must be of number of possible categories y may take.
	   num_layers -     number of layers, additional layer for the original x-es must be
	                    counted (and also included in layer_sizes as specified above)
	   train_data -     array of pointers, which point to separate x observations
	   ys -             elements of vector give labels to corresponding observations
	   num_iterations - max number of times the full sweep of Stochastic Gradient
	                    Descent should run when fitting the model
	   batch_size -     number of observations in a batch - each core will receive one batch
	   step_size -      currently fixed parameter for Stoch Grad Desc,
	                    TO DO will be changed in later versions
	   output_weigts -  fitted weights will go here
	   output_biases -  fitted biases will go here
	   nrow -           number of x observations
	   ncol -           dimensionality of x
	   transf... -      if 1 then softmax for last layer and cross entropy loss,
	                    else sigmoid for all layers and squared error loss

    Fits neural network using backpropagation algorithm (and batch updates)
    for arbitrary number of layers and units in layers. Currently supports
    only classification problems.
  */

  
  // set up the random number generator
  const gsl_rng_type* T;
  gsl_rng* r;
  gsl_rng_env_setup();
  T = gsl_rng_default;
  r = gsl_rng_alloc(T);
  gsl_rng_set(r, time(NULL));

  // initialize a list of vectors with biases
  // and a list of matrices with weights
  gsl_vector* biases[num_layers-1];
  gsl_matrix* weights[num_layers-1];

  gsl_vector* biases_momentum[num_layers-1];
  gsl_matrix* weights_momentum[num_layers-1];

   // no biases in the first layer, so start from (layer_sizes + 1)
  init_bias_object(biases, (layer_sizes+1),num_layers-1);
  init_weight_object(weights, layer_sizes,num_layers);

  init_bias_object(biases_momentum, (layer_sizes+1),num_layers-1);
  init_weight_object(weights_momentum, layer_sizes,num_layers);

  // set initial values
  randomize_bias(biases, (layer_sizes+1), num_layers-1, r);
  randomize_weight(weights, layer_sizes, num_layers-1, r);

  // all auxiliary parameters are stored here:
  par p;
  
  // set parameters that will be passed around:
  p.biases = biases;
  p.weights = weights;
  p.biases_momentum = biases_momentum;
  p.weights_momentum = weights_momentum;
  p.contract_momentum = 0.9;
  
  p.layer_sizes = layer_sizes;
  p.num_layers = num_layers;
  p.batch_number = core_num;
  p.batch_size = nrow / core_num;
  p.step_size = step_size;
  p.r = r;
  p.trans = sigmoid;
  p.trans_prime = sigmoid_prime;
  p.trans_final_prime = sigmoid_prime;
  p.penalty = penalty;
  p.transformation_type = transformation_type;
  p.nrow = nrow;
  p.ncol = ncol;
  if (transformation_type == 1){
    p.trans_final = softmax;
    p.cost = softmax_cost;
    p.cost_prime = softmax_prime;
  }
  else{
    p.trans_final = sigmoid;
    p.cost = squared_error_cost;
    p.cost_prime = softmax_prime;
  }

  // perform Batch updates num_iterations many times
  for (int i = 0; i < num_iterations; i++){
    p.total_cost = 0.0;
    BatchUpdate(train_data, ys, &p);
    // update the loss function history
    cost_hist[i] = p.total_cost + cost_regul_term(&p);
  }

  // copy the results to the output matrices and vectors
  for (int i = 0; i < num_layers-1; i++){
    gsl_vector_memcpy(output_biases[i], p.biases[i]);
    gsl_matrix_memcpy(output_weights[i], p.weights[i]);
  }

  destroy_parameters(&p);
} 


void BatchUpdate(gsl_vector* train_data[], gsl_vector* ys, par* p){
  /*
    performs one full sweep of Stochastic Gradient Descent (actually batch updates now):
    shuffles the data, splits into batches, (TO DO: sends each batch to separate core)
    performs forward propagation and backward propagation for each observation within
    batch, updates biases and weights vectors and repeats for as many batches as can
    be fit in the nrow without any repeats
  */

  // shuffle the data:
  int m = p->nrow;
  int indices[m];
  for (int i = 0; i < m; i++){
    indices[i] = i;
  }
  gsl_ran_shuffle(p->r,indices, m, sizeof(int));
  
  gsl_vector* bias_updates[p->num_layers-1];
  gsl_matrix* weight_updates[p->num_layers-1];

  init_bias_object(bias_updates,(p->layer_sizes+1), p->num_layers-1);
  init_weight_object(weight_updates,p->layer_sizes, p->num_layers);

  omp_set_dynamic(0);
  omp_set_num_threads(p->batch_number);
  #pragma omp parallel
  {
    //for (int i = 0; i < batch_number; i++){                                  // uncomment for the serial
    // SGD update for a batch, this will be done by each core separately:
    // ----------------------------
    
    // initialise parameters that will be shared in each core
    int i = omp_get_thread_num();
    
    par_c q;
    q.total_cost = 0;
    // init_parameters_core(&q, p);
    int t = p->num_layers;
    int* ls = p->layer_sizes;
    gsl_vector* gradient_biases[t-1];
    gsl_matrix* gradient_weights[t-1];
    gsl_vector* z[t];
    gsl_vector* transf_x[t];
    gsl_vector* delta[t-1];
  
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
    for (int j = 0; j < p->batch_size; j++){
      q.x = train_data[i * p->batch_size + j];
      q.y = (int) gsl_vector_get(ys,i * p->batch_size + j);
      forward(&q,p);
      backpropagation(&q,p);
    }

    #pragma omp critical
    {
    p->total_cost += q.total_cost/m;
    for (int j = 0; j < p->num_layers-1; j++){
      gsl_vector_add(bias_updates[j],q.gradient_biases[j]);
      gsl_matrix_add(weight_updates[j], q.gradient_weights[j]);
    }
    //printf("\nFirst element: %g, thread number: %d\n\n", gsl_matrix_get(bias_updates[0], 0,0), omp_get_thread_num());
    }
    destroy_parameters_core(&q,p);
    //}                                                                                // uncomment for the serial
  }
  double learning_rate = -p->step_size;
  regularisation(p, bias_updates, weight_updates);
  momentum_update(p, bias_updates, weight_updates, learning_rate);
  for (int i = 0; i < p->num_layers-1; i++){
    gsl_vector_free(bias_updates[i]);
    gsl_matrix_free(weight_updates[i]);
  }
}

void regularisation(par* p, gsl_vector** biases, gsl_matrix** weights)
{
  for (int i = 0; i < p->num_layers-1; i++){
    gsl_blas_daxpy(p->penalty, p->biases[i], biases[i]);

    gsl_matrix* temp = gsl_matrix_alloc((*weights[i]).size1,(*weights[i]).size2);
    gsl_matrix_memcpy(temp, p->weights[i]);
    gsl_matrix_scale(temp, -p->penalty);
    gsl_matrix_add(weights[i],temp);
    gsl_matrix_free(temp);
  }

}

void momentum_update(par* p, gsl_vector** biases, gsl_matrix** weights, double step)
{
  double mu = p->contract_momentum;
  for (int i = 0; i < p->num_layers-1; i++){
    gsl_vector_scale(p->biases_momentum[i], mu);
    gsl_blas_daxpy(step, biases[i], p->biases_momentum[i]);
    gsl_vector_add(p->biases[i], p->biases_momentum[i]);

    gsl_matrix_scale(p->weights_momentum[i], mu);
    gsl_matrix_scale(weights[i],step);
    gsl_matrix_add(p->weights_momentum[i],weights[i]);
    gsl_matrix_add(p->weights[i], p->weights_momentum[i]);
  }
}

#endif //_NEURALNETS_
