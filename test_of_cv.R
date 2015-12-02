library("OxWaSPneuralnets", lib.loc = "~/R")

gen_data = function(j, n=50){
  r = seq(0, 1, length.out = n)
  theta = seq(j*4, (j+1)*4, length.out = n) + rnorm(n, sd=0.2)
  x = r * cos(theta)
  y = r * sin(theta)
  return(data.frame(x1 = x, x2 = y, labels = j))
}

df = rbind(gen_data(0), gen_data(1), gen_data(2))
df = df[sample(1:nrow(df)), ]

X = cbind(df$x1, df$x2)
y = df$labels

plot(X, col=y+1, pch=16)

rand_seq = sample(1:nrow(X),size=nrow(X),replace=FALSE)

X_rand = X[rand_seq,]
Y_rand = y[rand_seq]

plot(X_rand, col=Y_rand+1, pch=16)
X_train = X_rand[1:100,]
Y_train = Y_rand[1:100]
X_test = X_rand[101:150,]
Y_test = Y_rand[101:150]

points(X_test, col=Y_test+1, pch=13)
### fit neural network

# First, do this on command line
# R CMD SHLIB cs231n.c -lm -lgsl -lgslcblas

layer_sizes = c(ncol(X_test),20,3)
num_layers = 3
num_iterations = 2000
core_size = 1
step_size = 0.001
penalty = c(0.0005,0.001,0.005,0.01,0.1)
fold = 5
n_classes = 3
out = .C("CvNN",
         X = as.vector(as.numeric(X_train)),y = as.double(Y_train),
         layer_sizes = as.integer(layer_sizes),
         num_layers = as.integer(num_layers),
         num_iterations = as.integer(num_iterations),
         Core_size = as.integer(core_size),
         step_size = as.double(step_size),
         nrow_train = as.integer(nrow(X_train)),
         ncol = as.integer(ncol(X_train)),
         penalty = as.double(penalty),
         penalty_size = as.integer(length(penalty)),
         trans_type = as.integer(1), Xtest = as.vector(as.numeric(X_test)),
         Ytest = as.double(Y_test),
         nrow_test = as.integer(nrow(X_test)), fold = as.integer(fold),
         pred_train = as.double(rep(0, nrow(train_X)*n_classes)),
         pred_test = as.integer(rep(0, nrow(test_X)))
         )


load("data/mnist.RData")

X = train$x[1:3000,]
y = train$y[1:3000]

rand_seq = sample(1:nrow(X),size=nrow(X),replace=FALSE)

X_rand = X[rand_seq,]
Y_rand = y[rand_seq]

X_train = X_rand[1:2500,]
Y_train = Y_rand[1:2500]

X_test = X_rand[2501:3000,]
Y_test = Y_rand[2501:3000]


layer_sizes = c(ncol(X_train),20,10)
num_layers = 3
num_iterations = 500
core_size = 8
step_size = 0.0001
penalty = c(0.0001,0.1)
fold = 5
n_classes=10
out = .C("CvNN",
         X = as.vector(as.numeric(X_train)),y = as.double(Y_train),
         layer_sizes = as.integer(layer_sizes),
         num_layers = as.integer(num_layers),
         num_iterations = as.integer(num_iterations),
         Core_size = as.integer(core_size),
         step_size = as.double(step_size),
         nrow_train = as.integer(nrow(X_train)),
         ncol = as.integer(ncol(X_train)),
         penalty = as.double(penalty),
         penalty_size = as.integer(length(penalty)),
         trans_type = as.integer(1), Xtest = as.vector(as.numeric(X_test)),
         Ytest = as.double(Y_test),
         nrow_test = as.integer(nrow(X_test)), fold = as.integer(fold),
         pred_train = as.double(rep(0, nrow(X_test)*n_classes)),
         pred_test = as.integer(rep(0, length(Y_test)))
         )

