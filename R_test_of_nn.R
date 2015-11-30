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


### fit neural network

# First, do this on command line
# R CMD SHLIB cs231n.c -lm -lgsl -lgslcblas



layer_sizes = c(ncol(X),20,3)
num_layers = 3
num_iterations = 2000
batch_size = nrow(X) /10
step_size = 0.1
penalty = 0.001



out = .C("nn",
         X = as.vector(as.numeric(X)),y = as.double(y),
         layer_sizes = as.integer(layer_sizes),
         num_layers = as.integer(num_layers),
         num_iterations = as.integer(num_iterations),
         batch_size = as.integer(batch_size),
         step_size = as.double(step_size),
         nrow = as.integer(nrow(X)),
         ncol = as.integer(ncol(X)),
         penalty = as.double(penalty),
         output = as.double(-1), output2 = as.double(rep(0,num_iterations)),
         trans_type = as.integer(2))

out$output
plot(out$output2, type ="l")

## mnist data
load("data/mnist.RData")

X = train$x[1:2000,]
y = train$y[1:2000]

layer_sizes = c(ncol(X),20,10)
num_layers = 3
num_iterations = 1000
batch_size = nrow(X)/10
step_size = 0.01
penalty = 0.001



out = .C("nn",
         X = as.vector(as.numeric(X)),y = as.double(y),
         layer_sizes = as.integer(layer_sizes),
         num_layers = as.integer(num_layers),
         num_iterations = as.integer(num_iterations),
         batch_size = as.integer(batch_size),
         step_size = as.double(step_size),
         nrow = as.integer(nrow(X)),
         ncol = as.integer(ncol(X)),
         penalty = as.double(penalty),
         output = as.double(-1),
         output2 = as.double(rep(0,num_iterations)),
         trans_type = as.integer(1))

out$output
plot(out$output2,type = "l")
















