#ifndef _FORWARD_
#define _FORWARD_

#include "globhead.h"

void forward(par_c* q, par* p) {
  /*
   Forward sweep for x in q. Scores: z=Wx+b and transformed
   inputs: trans(z) are stored along the way.
   */

  // for input layer we use convention: x = z = transf(z)
  gsl_vector_memcpy(q->transf_x[0],q->x);
  gsl_vector_memcpy(q->z[0],q->x);

  for (int i = 0; i < p->num_layers-1; i++){
    // compute and store z

    compute_score(q,p,i);

    if (i == p->num_layers-2){
      // final layer transformation
      p->trans_final(q->transf_x[i+1]);
      q->total_cost += p->cost(q->transf_x[i+1], q->y);
    }
    else {
      // hidden layer transformation
      p->trans(q->transf_x[i+1]);
    }
  }
}

void compute_score(par_c* q, par* p, int i){
    //computes score z= Wx + b

  // compute Wx
  gsl_blas_dgemv(CblasNoTrans, 1, p->weights[i], q->transf_x[i], 0, q->transf_x[i+1]);
  // compute z = Wx + b
  gsl_vector_add(q->transf_x[i+1], p->biases[i]);
  gsl_vector_memcpy(q->z[i+1],q->transf_x[i+1]);
}


void compute_score_ispc(par_c* q, par* p, int ind){

  // vectorized version
  score_ispc(p->weights[ind]->data,
             p->biases[ind]->data,
             q->transf_x[ind]->data,
             q->transf_x[ind + 1]->data,
             p->weights[ind]->size1,
             p->weights[ind]->size2);

}



#endif // _FORWARD_
