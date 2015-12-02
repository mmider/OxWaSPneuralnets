#ifndef _SPLITFOLD_
#define _SPLITFOLD_

#include "globhead.h"

//Function for splitting data sets and giving the address to each fold.
void SplitFoldfunc(const gsl_matrix *TrainData, int fold, int* rand_seq, gsl_matrix** SubTrain){
 //*SubTrain = gsl_matrix_alloc(SizeGroup,Features);
 int N_obs = TrainData->size1;
 int Features = TrainData->size2;
// printf("%d\n\n",Features);
 int SizeGroup = N_obs/fold;
// printf("%d\n\n",SizeGroup);

//gsl_matrix *SubTrain[fold];
//gsl_matrix** SubTrain;
//gsl_matrix** SubTrain = malloc(sizeof(*gsl_matrix)*fold);
//gsl_matrix* Y;
//Y = gsl_matrix_alloc(SizeGroup, Features);
//printf("%d\n", sizeof(Y));
//gsl_matrix** SubTrain; //Create double pointer to hold 2d matrix
//printf("-----------------\n");
//printf("%d\n\n",fold);

//printf("-----------------\n");
//fold = (size_t) fold;
//printf("-----------------\n");
//*SubTrain = malloc(fold * sizeof(Y));
//printf("-----------------\n\n\n");
//printf("%d\n", *SubTrain);
//printf("%d\n", sizeof(Y));
/*for (int i = 0; i < fold; i++)
{
    SubTrain[i] = malloc(sizeof(*Y));
    printf("%d\n", i);
}
*/

//printf("-----------------\n");
//gsl_matrix_free(Y);
  for (int t = 0; t < fold; t++){
    gsl_matrix* K;
    K = gsl_matrix_alloc(SizeGroup, Features);
 //  printf("fold %d\n", t);
   int h = 0;
     for (int i = 0; i < N_obs; i++){
   //  printf("obs %d\n", i);
       if (rand_seq[i] == t){
          for (int s = 0; s < Features; s++){
   //        printf("feat %d\n", s);
   //        printf("setentryrow %d\n", h);
           double element_value;
           element_value = gsl_matrix_get(TrainData, i, s);
    //       printf("element %G\n", element_value);
           gsl_matrix_set(K, h, s, element_value);
          }
           h = h + 1;
        }
     printf("----------\n");
     }
  SubTrain[t] = K;
  }
}



#endif // _SPLITFOLD_

