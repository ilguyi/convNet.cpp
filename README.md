# Convolutional neural network for multi-class classification
implement with backpropagation using both Boost and Armadillo library in C++

# Author
Il Gu Yi


# Usage
- execute: ./conv train.d test.d parameter.d
- argument 1: train data (optional including validation data)
- argument 2: test data
- argument 3: parameters data
- DATA: MNIST (partial data) 


# What program can do (version 0.1) (2015. 09. 03.)
- 4 types of layers
 - Convolutional layer, Pooling layer, Fully Connected layer, Softmax layer
- 3 types of activation functions
-- Sigmoid, Tanh, ReLU
- Validation data (extracting randomly from train data)
- Two cost function (cross entropy or quadratic error)
- Apply L2-regularization (weight decay)
- Adjust mini-batch size
- 2 types of momemtum
-- ordinary momemtum, Nesterov momemtum


# Requirements
- I use the random number generator mt19937 from Boost library
for weights and bias initialization and stochastic gradient descent.
- I implement my program using Armadillo linear algebra library in C++
for various calculation based on matrix and vector.


## Version 0.1 (2015. 09. 03.)
- 4 types of layers
-- Convolutional layer, Pooling layer, Fully Connected layer, Softmax layer
- 3 types of activation functions
-- Sigmoid, Tanh, ReLU
- Validation data (extracting randomly from train data)
- Two cost function (cross entropy or quadratic error)
- Apply L2-regularization (weight decay)
- Adjust mini-batch size
- 2 types of momemtum
-- ordinary momemtum, Nesterov momemtum


