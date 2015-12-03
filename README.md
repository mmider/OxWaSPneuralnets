# OxWaSPneuralnets

This R package for fitting neural networks was written as a project for OxWaSP module 4 by Giuseppe Di Benedetto, Leon Law, Kaspar Märtens, and Marcin Mider.
The vignette for the package can be found [here](https://github.com/mmider/OxWaSPneuralnets/blob/master/vignettes/NeuralNets.Rmd).

To use our package, install it as follows

```R 
devtools::install_github("mmider/OxWaSPneuralnets")
```

Load the package

```R 
library(OxWaSPneuralnets)
```

and see the help files `?fit_neural_network` or `?CV_neural_network`. The following minimal example fits a neural network with one hidden layer

```R 
data(mnist)
# Pick only first 1000 data points (for speed)
res = fit_neural_network(train$x[1:1000, ], train$y[1:1000], test$x[1:250, ], test$y[1:250], n_iterations = 1000, step_size = 0.0001)
plot(res)
res
```
