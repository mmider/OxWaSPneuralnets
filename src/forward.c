#ifndef _FORWARD_
#define _FORWARD_

#include "globhead.h"

void forward(par_c* q, par* p) {
  /*
    Takes a single x and performs a single forward sweep for a given
    state of neural network. Stores the z=Wx+b for each layer in the
    z "history array" and stores sigmoid(Wx+b) (or possibly softmax(Wx+b)
    if softmax == true for the last layer) for each layer in the transf_x
  */
  
  // the first layer is sort of artificial, it is just defined for notational
  // convenience to contain the orignial x observation
  gsl_vector_memcpy((*q).transf_x[0],(*q).x);
  gsl_vector_memcpy((*q).z[0],(*q).x);
  
  for (int i = 0; i <(*p).num_layers-1; i++){
    // compute Wx
    gsl_blas_dgemv(CblasNoTrans, 1, (*p).weights[i], (*q).transf_x[i], 0, (*q).transf_x[i+1]);
    // update to z=Wx+b and store
    gsl_vector_add((*q).transf_x[i+1], (*p).biases[i]);
    gsl_vector_memcpy((*q).z[i+1],(*q).transf_x[i+1]);
    // update to sigmoid(z) or softmax(z)
    if (i == (*p).num_layers-2){
      (*p).trans_final((*q).transf_x[i+1]);
      (*q).total_cost += (*p).cost((*q).transf_x[i+1], (*q).y);
      // Rprintf("%g\n", (*q).total_cost);
    }
    else
      (*p).trans((*q).transf_x[i+1]);
  }
}

#endif // _FORWARD_
