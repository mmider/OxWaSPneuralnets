#ifndef _CVNEURAL2_
#define _CVNEURAL2_
#include "globhead.h"

void CrossVal(const gsl_matrix* XTrainData, const gsl_matrix* YTrainData, const gsl_matrix* XTestData,
              const gsl_matrix* YTestData, const int FOLD, const double* Lambda, const int sizelambda, const int* layer_sizes,  const int num_layers,
              const int num_iterations, const int core_size, const double step_size, const int trans_type, double* probs_test, int* predicted_test)
{
  int* layer_sizes_2 = layer_sizes;
  int ncat = layer_sizes_2[num_layers-1];
  //int layer_sizes_2[3]={784,20,10};
  int N_obs = XTrainData->size1;
  int YFeatures = YTrainData->size2;
  int XFeatures = XTrainData->size2;
  int GroupSize = N_obs/FOLD;
  int Nlambda = sizelambda;
  int N_obs_test = XTestData->size1;
  int* seq_fold;
  seq_fold = rand_fold(N_obs,FOLD);
  /*for (int i = 0; i < N_obs; i++){
    printf("%d\n",seq_fold[i]);
  }*/

  gsl_matrix* Xfolds[FOLD];
  for (int d = 0; d < FOLD; d++)
    Xfolds[d] = gsl_matrix_alloc(GroupSize,XFeatures);

  gsl_matrix* Yfolds[FOLD];
  for (int d = 0; d < FOLD; d++)
    Yfolds[d] = gsl_matrix_alloc(GroupSize,YFeatures);

  SplitFoldfunc(XTrainData, FOLD, seq_fold, Xfolds);
  SplitFoldfunc(YTrainData, FOLD, seq_fold, Yfolds);
/*
  for (int ss = 0; ss < GroupSize; ss++)
    printf( "Fold1 %G, %G\n",gsl_matrix_get(Xfolds[0],ss,0),gsl_matrix_get(Xfolds[0],ss,1));

  for (int ss = 0; ss < GroupSize; ss++)
    printf( "Fold2 %G, %G\n",gsl_matrix_get(Xfolds[1],ss,0),gsl_matrix_get(Xfolds[1],ss,1));

  for (int ss = 0; ss < GroupSize; ss++)
    printf( "Fold3 %G, %G\n",gsl_matrix_get(Xfolds[2],ss,0),gsl_matrix_get(Xfolds[2],ss,1));

  for (int ss = 0; ss < GroupSize; ss++)
    printf( "Fold4 %G, %G\n",gsl_matrix_get(Xfolds[3],ss,0),gsl_matrix_get(Xfolds[3],ss,1));

  for (int ss = 0; ss < GroupSize; ss++)
    printf( "Fold5 %G, %G\n",gsl_matrix_get(Xfolds[4],ss,0),gsl_matrix_get(Xfolds[4],ss,1));


  for (int ss = 0; ss < GroupSize; ss++)
    printf( "Fold1 %G \n",gsl_matrix_get(Yfolds[0],ss,0));

  for (int ss = 0; ss < GroupSize; ss++)
    printf( "Fold2 %G \n",gsl_matrix_get(Yfolds[1],ss,0));

  for (int ss = 0; ss < GroupSize; ss++)
    printf( "Fold3 %G, \n",gsl_matrix_get(Yfolds[2],ss,0));

  for (int ss = 0; ss < GroupSize; ss++)
    printf( "Fold4 %G, \n",gsl_matrix_get(Yfolds[3],ss,0));

  for (int ss = 0; ss < GroupSize; ss++)
    printf( "Fold5 %G, \n",gsl_matrix_get(Yfolds[4],ss,0));
*/
  gsl_matrix* CvTrainX[FOLD];
  for (int d = 0; d < FOLD; d++)
    CvTrainX[d] = gsl_matrix_calloc(GroupSize*(FOLD-1), XFeatures);

  gsl_matrix* CvTrainY[FOLD];
  for (int d = 0; d < FOLD; d++)
    CvTrainY[d] = gsl_matrix_calloc(GroupSize*(FOLD-1), YFeatures);

  combinefold(Xfolds, Yfolds, N_obs, FOLD, GroupSize, XFeatures, YFeatures, CvTrainX, CvTrainY);
/*
  for (int ss = 0; ss < N_obs-GroupSize; ss++)
    printf( "Group1 %G, %G\n",gsl_matrix_get(CvTrainX[0],ss,0),gsl_matrix_get(CvTrainX[0],ss,1));

  for (int ss = 0; ss < N_obs-GroupSize; ss++)
    printf( "GG2 %G, %G\n",gsl_matrix_get(CvTrainX[1],ss,0),gsl_matrix_get(CvTrainX[1],ss,1));

  for (int ss = 0; ss < N_obs-GroupSize; ss++)
    printf( "GG3 %G, %G\n",gsl_matrix_get(CvTrainX[2],ss,0),gsl_matrix_get(CvTrainX[2],ss,1));

  for (int ss = 0; ss < N_obs-GroupSize; ss++)
    printf( "GG4 %G, %G\n",gsl_matrix_get(CvTrainX[3],ss,0),gsl_matrix_get(CvTrainX[3],ss,1));

  for (int ss = 0; ss < N_obs-GroupSize; ss++)
    printf( "G5 %G, %G\n",gsl_matrix_get(CvTrainX[4],ss,0),gsl_matrix_get(CvTrainX[4],ss,1));
*/
  /*
  for (int ss = 0; ss < N_obs-GroupSize; ss++)
  Rprintf( "Group1 %G, \n",gsl_matrix_get(CvTrainY[0],ss,0));

  for (int ss = 0; ss < N_obs-GroupSize; ss++)
  Rprintf( "GG2 %G, \n",gsl_matrix_get(CvTrainY[1],ss,0));

  for (int ss = 0; ss < N_obs-GroupSize; ss++)
  Rprintf( "GG3 %G \n",gsl_matrix_get(CvTrainY[2],ss,0));

  for (int ss = 0; ss < N_obs-GroupSize; ss++)
  Rprintf( "GG4 %G, \n",gsl_matrix_get(CvTrainY[3],ss,0));

  for (int ss = 0; ss < N_obs-GroupSize; ss++)
  Rprintf( "G5 %G, \n",gsl_matrix_get(CvTrainY[4],ss,0));
  */
  gsl_vector* results_lambda;
  results_lambda = gsl_vector_alloc((size_t) Nlambda);
  double results[Nlambda][FOLD];
  //private(i, j, vec_cv_trainX, vec_cv_trainY, output_weights, output_biases, vec_cv_valX, vec_cv_valY) collapse(2)
#pragma omp parallel for collapse(2)
  for (int i = 0; i < Nlambda; i++){
    for (int j = 0; j < FOLD; j++){
      gsl_vector* vec_cv_trainX[N_obs-GroupSize];
      gsl_vector* vec_cv_trainY;
      gsl_matrix* output_weights[num_layers-1];
      gsl_vector* output_biases[num_layers-1];
      gsl_vector* vec_cv_valX[GroupSize];
      gsl_vector* vec_cv_valY;
      // Rprintf("Lambda=%G\n", Lambda[i]);
      // Rprintf("fold not included = %d\n", j);
      //gsl_vector* vec_cv_trainX[N_obs-GroupSize];
      for (int u = 0; u < (N_obs-GroupSize); u++ ){
        vec_cv_trainX[u] = gsl_vector_alloc(XFeatures);
      }

      for (int c = 0; c < (N_obs-GroupSize); c++){
        gsl_matrix_get_row(vec_cv_trainX[c], CvTrainX[j], c);
      }

      //for (int a = 0; a < (N_obs-GroupSize); a++){
      //printf("%G %G\n",gsl_vector_get(vec_cv_trainX[a],0), gsl_vector_get(vec_cv_trainX[a],1));
      //printf("%d\n", a);
      //}

      //gsl_vector* vec_cv_trainY;
      vec_cv_trainY = gsl_vector_alloc(N_obs-GroupSize);
      gsl_matrix_get_col(vec_cv_trainY, CvTrainY[j], 0);

      //for (int y = 0; y < (N_obs-GroupSize); y++){
      //printf("%G\n",gsl_vector_get(vec_cv_trainY,y));
      //printf("%d\n",y);
      //}
      //Note that always Y will be 1 column, so well defined.

      //gsl_matrix* output_weights[num_layers-1];
      //gsl_vector* output_biases[num_layers-1];
      init_bias_object(output_biases, (layer_sizes_2+1), num_layers-1);
      init_weight_object(output_weights, layer_sizes_2, num_layers);
      //printf(" Lambda = %G\n",Lambda[i]);
      double cost_hist[num_iterations];
      NeuralNets(layer_sizes_2, num_layers, vec_cv_trainX, vec_cv_trainY, num_iterations, 1,
                  step_size, output_weights, output_biases, (N_obs-GroupSize), XFeatures, Lambda[i], cost_hist, trans_type);
      //gsl_vector* vec_cv_valX[GroupSize];
      for (int u = 0; u < (GroupSize); u++){
        vec_cv_valX[u] = gsl_vector_alloc(XFeatures);
      }
      for (int c = 0; c < GroupSize; c++){
        gsl_matrix_get_row(vec_cv_valX[c], Xfolds[j], c);
      //  for (int s = 0; s < (N_obs-GroupSize); s++){
        //  if((gsl_vector_get(vec_cv_valX[c],1))==(gsl_vector_get(vec_cv_trainX[s],1))){
        //    printf("ERROR!!!!\n%G",gsl_vector_get(vec_cv_valX[c],1));
        //  }
       // }
      }
      //gsl_vector* vec_cv_valY;
      vec_cv_valY = gsl_vector_alloc(GroupSize);
      gsl_matrix_get_col(vec_cv_valY, Yfolds[j], 0);

      double probs[(GroupSize*ncat)];
      int predicted_val[GroupSize];

      for (int d = 0; d < (GroupSize*ncat); d++)
      probs[d] = 0;

      for (int s = 0; s < GroupSize; s++)
      predicted_val[s] = 0;

      results[i][j] = evaluate_results(vec_cv_valX, vec_cv_valY, output_biases, output_weights, GroupSize,ncat, num_layers, layer_sizes_2, trans_type, probs, predicted_val);
      /*
      gsl_vector* vec_cv_testX[N_obs_test];
      for (int u = 0; u < (N_obs_test); u++){
        vec_cv_testX[u] = gsl_vector_alloc(XFeatures);
      }
      for (int c = 0; c < N_obs_test; c++){
        gsl_matrix_get_row(vec_cv_testX[c], XTestData, c);
      }
      */
      //gsl_vector* vec_cv_testY;
      //vec_cv_testY = gsl_vector_alloc(N_obs_test);
      //gsl_matrix_get_col(vec_cv_testY, YTestData, 0);

      //for (int ss = 0; ss < N_obs_test; ss++)
      //  Rprintf( "GG3 %G\n",gsl_vector_get(vec_cv_testY, ss));

      //Note that always Y will be 1 column, so well defined.
      //double success_test_check = correct_guesses(vec_cv_testX, vec_cv_testY, output_biases, output_weights, N_obs_test, num_layers, layer_sizes_2);
      //printf("Check = %G\n", success_test_check);
      //  gsl_vector* vec_cv_valX[GroupSize];
      //printf("\n-------------------------------\n Lambda = %G \n i = %d, j = %d \n", Lambda[i] ,i , j);
      //printf("Result = %G \n Thread = %d\n--------------------------------\n",results[i][j],omp_get_thread_num());
      gsl_vector_free(vec_cv_valY);
      for (int u = 0; u < (GroupSize); u++){
        gsl_vector_free(vec_cv_valX[u]);
      }
      gsl_vector_free(vec_cv_trainY);
      for (int u = 0; u < (GroupSize); u++){
        gsl_vector_free(vec_cv_trainX[u]);
      }
    }
  }

  //gsl_vector* results_lambda;
  //results_lambda = gsl_vector_alloc((size_t) Nlambda);
  double results_mean_fold[Nlambda];
  for (int w = 0; w < Nlambda; w++)
    results_mean_fold[w] = 0;


  for (int s = 0; s < Nlambda ; s++){
    for (int m = 0; m < FOLD ; m++){
      printf("Lambda = %G, Result = %G\n",Lambda[s], results[s][m]);
    }
  }

  for (int s = 0; s < Nlambda ; s++){
    for (int m = 0; m < FOLD ; m++){
      results_mean_fold[s] = results[s][m]+ results_mean_fold[s];
    }
    gsl_vector_set(results_lambda, s, results_mean_fold[s]/(FOLD));
  }

  for (int s = 0; s < Nlambda ; s++){
    printf("Lambda = %G, Success = %G\n", Lambda[s], gsl_vector_get(results_lambda, s));
  }
  int OptimalLambda_index = gsl_vector_max_index(results_lambda);
  double Optimal_lambda = Lambda[OptimalLambda_index];
  gsl_vector_free(results_lambda);
  gsl_matrix* output_weights_all[num_layers-1];
  gsl_vector* output_biases_all[num_layers-1];
  init_bias_object(output_biases_all, (layer_sizes_2+1), num_layers-1);
  init_weight_object(output_weights_all, layer_sizes_2, num_layers);

  gsl_vector* vec_cv_trainX_all[N_obs];
  for (int u = 0; u < (N_obs); u++){
    vec_cv_trainX_all[u] = gsl_vector_alloc(XFeatures);
  }

  for (int c = 0; c < N_obs; c++){
    gsl_matrix_get_row(vec_cv_trainX_all[c], XTrainData, c);
  }

  gsl_vector* vec_cv_trainY_all;
  vec_cv_trainY_all = gsl_vector_alloc(N_obs);
  gsl_matrix_get_col(vec_cv_trainY_all, YTrainData, 0);

 // for (int ss = 0; ss < N_obs; ss++)
  //  printf( "GG3 %G\n",gsl_vector_get(vec_cv_trainY_all,ss));

  printf("Optimal Lambda = %G\n", Optimal_lambda);

  double cost_hist_test[num_iterations];
  NeuralNets(layer_sizes_2, num_layers, vec_cv_trainX_all, vec_cv_trainY_all, num_iterations, core_size,
              step_size, output_weights_all, output_biases_all, N_obs, XFeatures, Optimal_lambda, cost_hist_test, trans_type);
  gsl_vector_free(vec_cv_trainY_all);
  for (int u = 0; u < (N_obs); u++){
    gsl_vector_free(vec_cv_trainX_all[u]);
  }
  //printf("Optimal Lambda = %G\n", Optimal_lambda);

  gsl_vector* vec_cv_testX[N_obs_test];
  for (int u = 0; u < (N_obs_test); u++){
    vec_cv_testX[u] = gsl_vector_alloc(XFeatures);
  }
  for (int c = 0; c < N_obs_test; c++){
    gsl_matrix_get_row(vec_cv_testX[c], XTestData, c);
  }

  gsl_vector* vec_cv_testY;
  vec_cv_testY = gsl_vector_alloc(N_obs_test);
  gsl_matrix_get_col(vec_cv_testY, YTestData, 0);

 // for (int ss = 0; ss < N_obs_test; ss++)
  //  printf( "GG %G\n",gsl_vector_get(vec_cv_testY, ss));

  //Note that always Y will be 1 column, so well defined.

  double success_test = evaluate_results(vec_cv_testX, vec_cv_testY, output_biases_all, output_weights_all, N_obs_test,ncat, num_layers, layer_sizes_2, trans_type, probs_test, predicted_test);
  gsl_vector_free(vec_cv_testY);
  for (int u = 0; u < (N_obs_test); u++){
    gsl_vector_free(vec_cv_testX[u]);
  }
  /*gsl_vector* vec_cv_valX[GroupSize];
   for (int u = 0; u < (GroupSize); u++){
   vec_cv_valX[u] = gsl_vector_alloc(XFeatures);
   }
   for (int c = 0; c < GroupSize; c++){
   gsl_matrix_get_row(vec_cv_valX[c], Xfolds[0], c);
   }
   gsl_vector* vec_cv_valY;
   vec_cv_valY = gsl_vector_alloc(GroupSize);
   gsl_matrix_get_col(vec_cv_valY, Yfolds[0], 0);
   double success_test = correct_guesses(vec_cv_valX, vec_cv_valY, output_biases_all, output_weights_all, GroupSize, num_layers, layer_sizes);
   */
  printf("Test Success = %G \n",success_test);
}

#endif // _CVNEURAL2_

















































