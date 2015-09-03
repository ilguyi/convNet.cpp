/***********************************************************
 * Convolutional Neural networks for multi-class classification
 * some typedef namespace
 *
 * 2015. 08. 25.
 * modified 2015. 09. 03.
 * by Il Gu Yi
***********************************************************/

#ifndef CONV_NEURAL_NETWORK_DEF_H
#define CONV_NEURAL_NETWORK_DEF_H

#include <armadillo>
using namespace std;


namespace convdef {


typedef arma::mat Weight;
typedef arma::vec Bias;
typedef arma::field<arma::cube> Weights;

typedef arma::vec Vector;
typedef arma::mat Matrix;
typedef arma::cube Cube;


typedef enum {
    Conv,
    Pool,
    FC,
    Soft,
} Layer_Type;



typedef enum {
    //  using cross entropy cost function
    //  C = target * log activation + (1 - target) * log (1 - activation) 
    CrossEntropy,
    //  using quadratic cost function C = (target - activation)^2 / 2
    Quadratic,
} CostFunction_Type;


typedef enum {
    Sigmoid,
    Tanh,
    ReLU,
    Softmax,
    None,
} ActivationFuntion_Type;




}





#endif
