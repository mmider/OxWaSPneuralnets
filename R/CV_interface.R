#' Fit neural network, with parameter tuning via cross-validation
#'
#'@param train_X Matrix of training data (data points in rows, features in columns)
#'@param train_y Vector of labels for training data (these have to be integers from 0 to n_classes - 1)
#'@param test_X Matrix of test data
#'@param test_y Vector of labels for test data
#'@param n_hidden_layers Number of hidden layers in the neural network
#'@param hidden_layer_sizes Vector containing the number of neurons in each hidden layer
#'@param n_iterations The number of iterations for fitting the neural network
#'@param step_size The step size for updating parameters at each iteration
#'@param lambdas Vector of regularisation parameters (cross-validation is carried out over these)
#'@param n_cores The number of parallel cores
#'
#'@return List containing the following elements: \code{train_acc}, \code{test_acc}, \code{loss_over_time},
#'\code{prob_train}, \code{prob_test}, \code{pred_train}, \code{pred_test}
#'
#'@examples
#'
#' data(toy_data)
#' plot(toy_train$X, col=toy_train$y+1, pch=16)
#' res = CV_neural_network(toy_train$X, toy_train$y, toy_test$X, toy_test$y, n_iterations = 1000, step_size = 0.001)
#' res
#' # Confusion matrix for test data
#' table(res$pred_test, toy_test$y)
#'
#'\dontrun{
#' data(mnist)
#' # Pick only first 1000 data points (for speed)
#' res = CV_neural_network(train$x[1:500, ], train$y[1:500], test$x[1:250, ], test$y[1:250], n_iterations = 100, step_size = 0.0001)
#' table(res$pred_test, test$y[1:250])
#' }
#'@useDynLib OxWaSPneuralnets, .registration=TRUE
#'@export
#'
CV_neural_network = function(train_X, train_y, test_X, test_y,
                              n_hidden_layers = 1,
                              hidden_layer_sizes = c(20),
                              n_iterations = 100,
                              step_size = 0.01,
                              lambdas = c(0.0005,0.001,0.005,0.01,0.1),
                              n_folds = 5,
                              n_cores = 1){

  if(n_hidden_layers != length(hidden_layer_sizes)) stop("Misspecified hidden layer sizes!")
  if(nrow(train_X) != length(train_y)) stop("Dimensions of training data do not match")
  if(nrow(test_X) != length(test_y)) stop("Dimensions of test data do not match")

  n_layers = n_hidden_layers + 2
  n_classes = max(train_y) + 1
  layer_sizes = c(ncol(train_X), hidden_layer_sizes, n_classes)
  obj = .C("CvNN",
           X = as.vector(as.numeric(train_X)),
           y = as.double(train_y),
           layer_sizes = as.integer(layer_sizes),
           num_layers = as.integer(n_layers),
           n_iterations = as.integer(n_iterations),
           n_cores = as.integer(n_cores),
           step_size = as.double(step_size),
           nrow_train = as.integer(nrow(train_X)),
           ncol = as.integer(ncol(train_X)),
           lambdas = as.double(lambdas),
           len_lambda = as.integer(length(lambdas)),
           trans_type = as.integer(1),
           Xtest = as.vector(as.numeric(test_X)),
           Ytest = as.double(test_y),
           nrow_test = as.integer(nrow(test_X)),
           n_folds = as.integer(n_folds),
           pred_train = as.double(rep(0, nrow(train_X)*n_classes)),
           pred_test = as.integer(rep(0, nrow(test_X)))
  )
  out = list("pred_train" = obj$pred_train,
             "pred_test" = obj$pred_test)

  return(out)
}
