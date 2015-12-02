#ifndef _RANDPERM_
#define _RANDPERM_

#include "globhead.h"

// Make Random Permuatuation
int* randomperm11(int n)
{
    int* r = malloc(n * sizeof(int));
    for(int i=0;i<n;++i){
        r[i]=i+1;
    }
    for (int i = n - 1; i >= 0; --i){
        //generate a random number [0, n-1]
        int j = rand() % (i+1);
        //swap the last element with element at random index
        int temp = r[i];
        r[i] = r[j];
        r[j] = temp;
    }
    return r;
}

#endif // _RANDPERM_
