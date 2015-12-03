export void score_ispc(uniform double w[],
       	               uniform double b[],
		       uniform double x[],
		       uniform double res[],
		       uniform int nrow,
		       uniform int ncol) {

  for (int i = 0; i < nrow; i++) {
    double s = 0.0;
    foreach(j = 0 ... ncol) {
      s += w[i * ncol + j] * x[j];
    }
    res[i] = reduce_add(s) + b[i];
  }
}


export void updatebias_ispc(uniform double b[],
                            uniform double db[],
                            uniform double momentumb[],
                            uniform double out[],
                            uniform double mu,
                            uniform double lambda,
                            uniform double step,
                            uniform size_t n){
  // momentumb[i] = mu * momentumb[i] + step * (dbiases[i] + lambda*biases[i])
  foreach (i = 0 ... n) {
    out[i] = mu * momentumb[i] + step * (db[i] + lambda * b[i]);
  }
}

export void updateweights_ispc(uniform double W[],
                              uniform double dW[],
                              uniform double momentumW[],
                              uniform double out[],
                              uniform double mu,
                              uniform double lambda,
                              uniform double step,
                              uniform size_t nrow,
                              uniform size_t ncol){
  int index;
  // momentumb[i] = mu * momentumb[i] + step * (dbiases[i] + lambda*biases[i])
  for(int i=0; i<nrow; i++){
    foreach (j = 0 ... ncol) {
      // assume weights are ordered by row
      index = i*ncol + j;
      out[index] = mu * momentumW[index] + step * (dW[index] + lambda * W[index]);
    }
  }
}
