#ifndef _CVNNWRAP_
#define _CVNNWRAP_

#include "globhead.h"

void CvNN(double* train_data, double* ys, int* layer_sizes, int* num_layers, int* num_iterations,
	int* core_num, double* step_size, int* nrow, int* ncol, double* penalty, int* penalty_size,
	int* trans_type, double* test_data_x, double* test_data_y, int* nrow_test, int* fold, double* probs_test, int* predicted_test){
//Wrapper Function for the CrossVal Function, allocates data into necessary format
	gsl_matrix* X_train = gsl_matrix_alloc(*nrow,*ncol);
	for (int j = 0; j < *ncol; j++){
	 for (int i = 0; i < *nrow; i++){
	     gsl_matrix_set(X_train,i,j,train_data[i+j* *nrow]);
		 }
	 }

	 gsl_matrix* Y_train = gsl_matrix_alloc(*nrow,1);
	 	 for (int i = 0; i < *nrow; i++){
	 	     gsl_matrix_set(Y_train,i,0,ys[i]);
	 	 }

		 	gsl_matrix* X_test = gsl_matrix_alloc(*nrow_test,*ncol);
		 	for (int j = 0; j < *ncol; j++){
		 	 for (int i = 0; i < *nrow_test; i++){
		 	     gsl_matrix_set(X_test,i,j,test_data_x[i+j* *nrow_test]);
		 		 }
		 	 }

		 	 gsl_matrix* Y_test = gsl_matrix_alloc(*nrow_test,1);
		 	 	 for (int i = 0; i < *nrow_test; i++){
		 	 	     gsl_matrix_set(Y_test,i,0,test_data_y[i]);
		 	 	 }
 //Run Cross validation and Prints Results
	CrossVal(X_train, Y_train, X_test, Y_test,
	       *fold, penalty, *penalty_size, layer_sizes,
				 *num_layers,*num_iterations, *core_num, *step_size, *trans_type, probs_test, predicted_test);
 // Free memory allocation
				 gsl_matrix_free(X_train);
				 gsl_matrix_free(Y_train);
				 gsl_matrix_free(X_test);
				 gsl_matrix_free(Y_test);
}

#endif // _CVNNWRAP_
