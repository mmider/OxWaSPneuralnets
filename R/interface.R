#' Fit the neural network
#'
#'@param train_X Matrix of training data (data points in rows, features in columns)
#'@param train_y Vector of labels for training data (these have to be integers from 0 to n_classes - 1)
#'@param test_X Matrix of test data
#'@param test_y Vector of labels for test data
#'@param n_hidden_layers Number of hidden layers in the neural network
#'@param hidden_layer_sizes Vector containing the number of neurons in each hidden layer
#'@param n_iterations The number of iterations for fitting the neural network
#'@param step_size The step size for updating parameters at each iteration
#'@param lambda The regularisation parameter
#'@param core_num The number of parallel cores
#'
#'@return List containing elements \code{train_acc}, \code{test_acc}, \code{loss_over_time}
#'
#'@examples
#'\dontrun{
#' data(toy_data)
#' plot(train$X, col=train$y+1, pch=16)
#' res = fit_neural_network(train$X, train$y, test$X, test$y, n_iterations = 1000, step_size = 0.001)
#' plot(res)
#' res
#'
#'
#' data(mnist)
#' # Pick only first 1000 data points (for speed)
#' res = fit_neural_network(train$x[1:1000, ], train$y, test$x[1:500, ], test$y, n_iterations = 1000, step_size = 0.0001)
#' plot(res)
#' res
#' }
#'@export
#'
fit_neural_network = function(train_X, train_y, test_X, test_y,
                              n_hidden_layers = 1,
                              hidden_layer_sizes = c(20),
                              n_iterations = 100,
                              step_size = 0.01,
                              lambda = 0.001,
                              core_num = 8){

  if(n_hidden_layers != length(hidden_layer_sizes)) stop("Misspecified hidden layer sizes!")
  if(nrow(train_X) != length(train_y)) stop("Dimensions of training data do not match")
  if(nrow(test_X) != length(test_y)) stop("Dimensions of test data do not match")

  n_layers = n_hidden_layers + 2
  n_classes = max(train_y) + 1
  layer_sizes = c(ncol(train_X), hidden_layer_sizes, n_classes)
  obj = .C("nn", X = as.vector(as.numeric(train_X)),
           y = as.double(train_y),
           test_X = as.vector(as.numeric(test_X)),
           test_y = as.double(test_y),
           layer_sizes = as.integer(layer_sizes),
           n_layers = as.integer(n_layers),
           num_iterations = as.integer(n_iterations),
           core_num = as.integer(core_num),
           step_size = as.double(step_size),
           nrow = as.integer(nrow(train_X)),
           ncol = as.integer(ncol(train_X)),
           nrow_test = as.integer(nrow(test_X)),
           penalty = as.double(lambda),
           train_acc = as.double(-1),
           loss_over_time = as.double(rep(0,n_iterations)),
           test_acc = as.double(-1),
           prob_train = as.double(rep(0, nrow(train_X)*n_classes)),
           prob_test = as.double(rep(0, nrow(test_X)*n_classes)),
           pred_train = as.integer(rep(0, nrow(train_X))),
           pred_test = as.integer(rep(0, nrow(test_X))),
           trans_type = as.integer(1))
  out = list("train_acc" = obj$train_acc,
             "test_acc" = obj$test_acc,
             "loss_over_time" = obj$loss_over_time,
             "prob_train" = as.matrix(obj$prob_train, ncol=n_classes),
             "prob_test" = as.matrix(obj$prob_test, ncol=n_classes),
             "pred_train" = obj$pred_train,
             "pred_test" = obj$pred_test)
  class(out) = "OxWaSPNN"
  return(out)
}

#' Print a summary of the neural network
#'
#' @export
#'
print.OxWaSPNN = function(obj){
  s = sprintf("Train accuracy: %1.3f\nTest accuracy: %1.3f", obj$train_acc, obj$test_acc)
  cat(s)
}

#' Plot the history of loss function over the time
#'
#' @export
#'
plot.OxWaSPNN = function(obj){
  plot(obj$loss_over_time, type="l", main="Loss on training data", xlab="iterations", ylab="loss")
}
