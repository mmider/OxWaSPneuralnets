#ifndef _COMBINEFOLD_
#define _COMBINEFOLD_

#include "globhead.h"


void combinefold (gsl_matrix** foldX, gsl_matrix** foldY, int N_obs, int fold, int SizeGroup, int Features, int label_size, gsl_matrix** CvTrainX, gsl_matrix** CvTrainY){
     for (int j = 0; j < fold; j++){
       int count = 0;
       gsl_matrix_view FoldX[fold];
       gsl_matrix_view FoldY[fold];
         for (int f = 0; f < fold; f++){
             if (f == j) continue;
              FoldX[f] = gsl_matrix_submatrix(CvTrainX[j], SizeGroup*count, 0, SizeGroup, Features);
              FoldY[f] = gsl_matrix_submatrix(CvTrainY[j], SizeGroup*count, 0, SizeGroup, label_size);
              gsl_matrix_memcpy(&FoldX[f].matrix, foldX[f]);
              gsl_matrix_memcpy(&FoldY[f].matrix, foldY[f]);
              count = count + 1;
            }
      }
}

#endif // _COMBINEFOLD_
