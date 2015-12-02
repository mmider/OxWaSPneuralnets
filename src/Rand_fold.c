#ifndef _RAND_FOLD_
#define _RAND_FOLD_

#include "globhead.h"

int* rand_fold(const int N_obs, const int fold){
 int SizeGroup = N_obs/fold;
 printf("%d\n",SizeGroup);
 int *rand_seq = malloc(sizeof(int)*N_obs);
 srand(time(NULL));

  int* rand_seq_perm;
  int foldset = 0;
  rand_seq_perm = randomperm11(N_obs);
  for (int i = 0; i < N_obs; i++){
  printf("%d\n",rand_seq_perm[i]);
  }
  int begin = 1;
  int Total = SizeGroup;
  while (Total <= N_obs){
    for (int i = 0; i < N_obs; i++){
      if (rand_seq_perm[i] >= begin && rand_seq_perm[i] <= Total){
       rand_seq[i] = foldset;
      }
    }
   foldset = foldset + 1;
   begin = begin + SizeGroup;
   Total = Total + SizeGroup;
  }
  for (int i = 0; i < N_obs; i++){
  printf("%d\n\n",rand_seq[i]);
  }
  return(&rand_seq[0]);
}

#endif // _RAND_FOLD_
