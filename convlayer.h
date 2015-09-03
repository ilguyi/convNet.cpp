/***********************************************************
 * Convolutional Neural networks for multi-class classification
 * Layer class
 *
 * 2015. 08. 25.
 * modified 2015. 09. 03.
 * by Il Gu Yi
***********************************************************/

#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include <boost/random.hpp>
#include <armadillo>
#include <string>
#include "dataframe.h"
#include "convdef.h"
using namespace std;
using namespace df;
using namespace convdef;


namespace convlayer {



/***************************************************************
 * Layer class (parents class)
***************************************************************/
class Layer {
    public:
        Layer();
        
        virtual void Initialize_Layer(const string& layerName, const arma::ivec& input_dimension, const arma::ivec& output_dimension,
                const ActivationFuntion_Type& actFunc, const double& dropout) = 0;

        virtual void PrintWeight() const = 0;
        virtual void PrintBias() const = 0;
        virtual void WriteResults(const string& filename) const = 0;
        virtual void WriteResults(const string& filename, const bool& append) const = 0;

        Cube ReshapeVectortoCube(const Vector& input_vec, const unsigned& height, const unsigned& width, const unsigned& depth);
        Cube ReshapeVectortoCube(const arma::ivec& input_vec, const unsigned& height, const unsigned& width, const unsigned& depth);
        Vector ReshapeCubetoVector(const Cube& input_cube);

        virtual void InitializeDeltaParameters() = 0;

        virtual void Forward(const Cube& input_cube) = 0;

        virtual void Backward(const Layer& nextLayer) = 0;
        virtual void Backward(const arma::ivec& t, const CostFunction_Type& cost) = 0;

        Cube DerivativeSigmoid();
        Cube DerivativeTanh();
        Cube DerivativeReLU();

        virtual void Cumulation(const Cube& preLayer_act) = 0;


        virtual void WithoutMomentum(const double& learningRate, const double& regularization,
                const unsigned& N_train, const unsigned& minibatchSize) = 0;
        virtual void Momentum(const double& learningRate, const double& regularization,
                const double& momentum, const unsigned& N_train, const unsigned& minibatchSize) = 0;
        virtual void NesterovMomentum(const double& learningRate, const double& regularization,
                const double& momentum, const unsigned& N_train, const unsigned& minibatchSize) = 0;


    public:
        string layerName;
        Layer_Type layerType;
        ActivationFuntion_Type actFunc;
        double dropout;
        
        arma::ivec input_dimension;
        arma::ivec output_dimension;

        Weights weight;
        Bias bias;

        Cube summation;
        Cube activation;

        Cube delta;
        Weights delta_weight;
        Bias delta_bias;

        Weights velocity_weight;
        Bias velocity_bias;
};

Layer::Layer() {};


Cube Layer::ReshapeVectortoCube(const arma::ivec& input_vec, const unsigned& height, const unsigned& width, const unsigned& depth) {
    
    Cube input(height, width, depth);
    for (unsigned k=0; k<depth; k++)
        for (unsigned j=0; j<width; j++)
            for (unsigned i=0; i<height; i++)
//                input(i, j, k) = (double) input_vec(i + j*height+ k*height*width);
                input(i, j, k) = (double) input_vec(i*width + j + k*height*width);

    return input;
}

Cube Layer::ReshapeVectortoCube(const Vector& input_vec, const unsigned& height, const unsigned& width, const unsigned& depth) {
    
    Cube input(height, width, depth);
    for (unsigned k=0; k<depth; k++)
        for (unsigned j=0; j<width; j++)
            for (unsigned i=0; i<height; i++)
//                input(i, j, k) = input_vec(i + j*height+ k*height*width);
                input(i, j, k) = input_vec(i*width + j + k*height*width);

    return input;
}

Vector Layer::ReshapeCubetoVector(const Cube& input_cube) {
    
    unsigned height = input_cube.n_rows;
    unsigned width = input_cube.n_cols;
    unsigned depth = input_cube.n_slices;
    Vector input(height * width * depth);

    for (unsigned k=0; k<depth; k++)
        for (unsigned j=0; j<width; j++)
            for (unsigned i=0; i<height; i++)
//                input(i + j*height + k*height*width) = input_cube(i, j, k);
                input(i*width + j + k*height*width) = input_cube(i, j, k);

    return input;
}

Cube Layer::DerivativeSigmoid() {
    return this->activation % (1. - this->activation);
}

Cube Layer::DerivativeTanh() {
    return (1. + this->activation) % (1. - this->activation) * 0.5;
}

Cube Layer::DerivativeReLU() {
    Cube temp(this->output_dimension(0), this->output_dimension(1), this->output_dimension(2));
    for (unsigned k=0; k<this->summation.n_slices; k++)
        for (unsigned j=0; j<this->summation.n_cols; j++)
            for (unsigned i=0; i<this->summation.n_rows; i++)
                temp(i, j, k) = this->summation(i, j, k) > 0 ? 1. : 0.;

    return temp;
}








/***************************************************************
 * Convolution Layer
***************************************************************/
class ConvLayer : public Layer {
    public:
        ConvLayer();
        
        void Initialize_Layer(const string& layerName, const arma::ivec& input_dimension, const arma::ivec& filter_dimension,
                const ActivationFuntion_Type& actFunc, const double& dropout);
        
        void PrintWeight() const;
        void PrintBias() const;
        void WriteResults(const string& filename) const;
        void WriteResults(const string& filename, const bool& append) const;

        void InitializeDeltaParameters();

        void Forward(const Cube& input);
        void Convolution(const Cube& input, Cube& output);
        double ConvolutionOne(const Cube& input, const unsigned& indexl, const unsigned& indexm, const unsigned& indexn);
        void Activation();


        void Backward(const Layer& nextLayer);
        void Backward(const arma::ivec& t, const CostFunction_Type& cost);
        double DerivativeMaxPooling(const double& act, const double& next_act);

        void Cumulation(const Cube& preLayer_act);
        double DeltaConvolutionOne(const Cube& input, const Cube& output, unsigned& indexi, const unsigned& indexj, const unsigned& indexk, const unsigned& indexn);

        void WithoutMomentum(const double& learningRate, const double& regularization, const unsigned& N_train, const unsigned& minibatchSize);
        void Momentum(const double& learningRate, const double& regularization,
                const double& momentum, const unsigned& N_train, const unsigned& minibatchSize);
        void NesterovMomentum(const double& learningRate, const double& regularization,
                const double& momentum, const unsigned& N_train, const unsigned& minibatchSize);


    public:
        Bias bias;
        Bias delta_bias;
        Bias velocity_bias;

        arma::ivec filter_dimension;
};

ConvLayer::ConvLayer() {};

void ConvLayer::Initialize_Layer(const string& layerName, const arma::ivec& input_dimension, const arma::ivec& filter_dimension,
        const ActivationFuntion_Type& actFunc, const double& dropout) {

    this->layerName = layerName;
    this->layerType = Conv;
    this->actFunc = actFunc;
    this->dropout = 0.;

    this->input_dimension = input_dimension;
    this->filter_dimension = filter_dimension;

    if ( this->input_dimension.size() != 3 ) {
        cout << "Usage: Wrong input data dimension (height, width, # of input images) " << endl;
        exit(1);
    }
    if ( filter_dimension.size() != 3 ) {
        cout << "Usage: Wrong filter (local receptive field) dimension (height, width, # of feature maps) " << endl;
        exit(1);
    }

    //  weights dimension (local receptive field height, lrf width, input depth, lrf depth)
    this->weight.set_size(filter_dimension(2));
    for (unsigned n=0; n<filter_dimension(2); n++)
        this->weight(n).set_size(filter_dimension(0), filter_dimension(1), this->input_dimension(2));
    //  bias dimension (local receptive field depth)
    this->bias.set_size(filter_dimension(2));

    unsigned feature_maps_height = this->input_dimension(0) - filter_dimension(0) + 1;
    unsigned feature_maps_width = this->input_dimension(1) - filter_dimension(1) + 1;
    unsigned feature_maps_depth = filter_dimension(2);
    this->output_dimension.set_size(3);
    this->output_dimension(0) = feature_maps_height;
    this->output_dimension(1) = feature_maps_width;
    this->output_dimension(2) = feature_maps_depth;
    this->summation.set_size(feature_maps_height, feature_maps_width, feature_maps_depth);
    this->activation.set_size(feature_maps_height, feature_maps_width, feature_maps_depth);

    unsigned n_output = feature_maps_height * feature_maps_width * feature_maps_depth;
    this->delta.set_size(feature_maps_height, feature_maps_width, feature_maps_depth);
    this->delta_weight.set_size(feature_maps_depth);
    for (unsigned n=0; n<feature_maps_depth; n++)
        this->delta_weight(n).set_size(filter_dimension(0), filter_dimension(1), this->input_dimension(2));
    this->delta_bias.set_size(filter_dimension(2));

    this->velocity_weight.set_size(feature_maps_depth);
    for (unsigned n=0; n<feature_maps_depth; n++)
        this->velocity_weight(n).set_size(filter_dimension(0), filter_dimension(1), this->input_dimension(2));
    this->velocity_bias.set_size(filter_dimension(2));


    //  weight Initialize from Gaussian distribution
    double std_dev = 1. / sqrt(n_output);
    boost::random::normal_distribution<> normal_dist(0., std_dev);         //  Choose a distribution
    boost::random::variate_generator<boost::random::mt19937 &,
        boost::random::normal_distribution<> > nrnd(rng, normal_dist);      //  link the Generator to the distribution

    for (unsigned n=0; n<this->weight.size(); n++) {
        this->velocity_weight(n).zeros();
        for (unsigned k=0; k<this->weight(n).n_slices; k++)
            for (unsigned j=0; j<this->weight(n).n_cols; j++)
                for (unsigned i=0; i<this->weight(n).n_rows; i++)
                    this->weight(n)(i,j,k) = nrnd();
    }

    boost::random::normal_distribution<> normal_dist1(0., 1.);         //  Choose a distribution
    boost::random::variate_generator<boost::random::mt19937 &,
        boost::random::normal_distribution<> > nrnd1(rng, normal_dist1);      //  link the Generator to the distribution

    for (unsigned n=0; n<bias.size(); n++)
        this->bias(n) = nrnd1();
    this->velocity_bias.zeros();
}


void ConvLayer::PrintWeight() const {
//  cout.precision(10);
//  cout.setf(ios::fixed);

    cout << "################################" << endl;
    cout << layerName << endl;
    cout << "################################" << endl;
    for (unsigned n=0; n<this->weight.size(); n++) {
        string ment = "conv layer weight : ";
        stringstream ss;    ss << n+1;
        ment += ss.str();
        ment += " feature maps ";

        this->weight(n).raw_print(ment);
    }
}

void ConvLayer::PrintBias() const {
    this->bias.raw_print("conv layer bias");
    cout << endl;
}


void ConvLayer::WriteResults(const string& filename) const {
    ofstream fsave(filename.c_str());
    fsave.precision(10);
    fsave.setf(ios::fixed);

    fsave << "################################" << endl;
    fsave << layerName << endl;
    fsave << "################################" << endl;
    for (unsigned n=0; n<this->weight.size(); n++) {
        string ment = "conv layer weight : ";
        stringstream ss;    ss << n+1;
        ment += ss.str();
        ment += " feature maps ";

        this->weight(n).raw_print(fsave, ment);
    }
    
    this->bias.raw_print(fsave, "conv layer bias");
    fsave << endl;
}

void ConvLayer::WriteResults(const string& filename, const bool& append) const {
    ofstream fsave;
    if ( !append )
        fsave.open(filename.c_str());
    else
        fsave.open(filename.c_str(), fstream::out | fstream::app);
    fsave.precision(10);
    fsave.setf(ios::fixed);

    fsave << "################################" << endl;
    fsave << layerName << endl;
    fsave << "################################" << endl;
    for (unsigned n=0; n<this->weight.size(); n++) {
        string ment = "conv layer weight : ";
        stringstream ss;    ss << n+1;
        ment += ss.str();
        ment += " feature maps ";

        this->weight(n).raw_print(fsave, ment);
    }
    
    this->bias.raw_print(fsave, "conv layer bias");
    fsave << endl;
}


void ConvLayer::InitializeDeltaParameters() {

    for (unsigned n=0; n<this->delta_weight.size(); n++)
        this->delta_weight(n).zeros();
    this->delta_bias.zeros();
}



void ConvLayer::Forward(const Cube& input) {

    Convolution(input, this->summation);
    Activation();

/*    cout << layerName << endl;
    cout << "Forward Conv input " << endl;
    cout << input << endl;
    cout << "Forward Conv summation" << endl;
    cout << this->summation << endl;
    cout << "Forward Conv activation" << endl;
    cout << this->activation << endl;
    */
}


void ConvLayer::Convolution(const Cube& input, Cube& output) {

    for (unsigned n=0; n<this->output_dimension(2); n++) {
        for (unsigned m=0; m<this->output_dimension(1); m++)
            for (unsigned l=0; l<this->output_dimension(0); l++)
                output(l, m, n) = ConvolutionOne(input, l, m, n);
    }
}

double ConvLayer::ConvolutionOne(const Cube& input, const unsigned& indexl, const unsigned& indexm, const unsigned& indexn) {

    double sum = 0.;
    for (unsigned k=0; k<this->input_dimension(2); k++)
        for (unsigned j=0; j<filter_dimension(1); j++)
            for (unsigned i=0; i<filter_dimension(0); i++)
                sum += this->weight(indexn)(i, j, k) * input(i+indexl, j+indexm, k);

    return sum + bias(indexn);
}

void ConvLayer::Activation() {
    
    if ( actFunc == Sigmoid )
        this->activation = 1. / (1. + exp(-this->summation));
    else if ( actFunc == Tanh )
        this->activation = tanh(this->summation);
    else if ( actFunc == ReLU )
        for (unsigned k=0; k<this->summation.n_slices; k++)
            for (unsigned j=0; j<this->summation.n_cols; j++)
                for (unsigned i=0; i<this->summation.n_rows; i++)
                    this->activation(i, j, k) = this->summation(i, j, k) > 0 ? this->summation(i, j, k) : 0.;
}




void ConvLayer::Backward(const Layer& nextLayer) {

   if ( nextLayer.layerType == Pool ) {
        unsigned pooling_size_height = (unsigned) (this->output_dimension(0) / nextLayer.output_dimension(0));
        unsigned pooling_size_width = (unsigned) (this->output_dimension(1) / nextLayer.output_dimension(1));

        for (unsigned k=0; k<this->output_dimension(2); k++)
            for (unsigned j=0; j<this->output_dimension(1); j++)
                for (unsigned i=0; i<this->output_dimension(0); i++)
                    this->delta(i, j, k) = nextLayer.delta((unsigned) (i/pooling_size_height), (unsigned) (j/pooling_size_width), k)
                                        * DerivativeMaxPooling(this->activation(i, j, k),
                                            nextLayer.activation((unsigned) (i/pooling_size_height), (unsigned) (j/pooling_size_width), k));
        this->delta %= DerivativeSigmoid();
    }
    else if ( nextLayer.layerType == FC  ||  nextLayer.layerType == Soft ) {
        Vector nextLayer_delta = ReshapeCubetoVector(nextLayer.delta);
        this->delta = ReshapeVectortoCube((nextLayer_delta.t() * nextLayer.weight(0).slice(0)).t(),
            this->output_dimension(0), this->output_dimension(1), this->output_dimension(2));
    }
    else {
        cout << "Usage: Not yet implemented in ConvLayer : next Conv" << endl;
        exit(1);
    }

/*    cout << layerName << endl;
    cout << "Backward Conv nextLayer delta" << endl;
    cout << nextLayer.delta << endl;
    cout << "nextLayer weight" << endl;
    cout << nextLayer.weight << endl;
    cout << "Backward Conv delta" << endl;
    cout << this->delta << endl;
    */
}

void ConvLayer::Backward(const arma::ivec& t, const CostFunction_Type& cost) {
    cout << "Usage: No Backward function in ConvLayer" << endl;
    exit(1);
}


double ConvLayer::DerivativeMaxPooling(const double& act, const double& next_act) {
    if ( act != next_act )
        return 0.;
    return 1.;
}


void ConvLayer::Cumulation(const Cube& preLayer_act) {

    for (unsigned n=0; n<this->output_dimension(2); n++)
        for (unsigned k=0; k<this->input_dimension(2); k++)
            for (unsigned j=0; j<filter_dimension(1); j++)
                for (unsigned i=0; i<filter_dimension(0); i++)
                    this->delta_weight(n)(i, j, k) += DeltaConvolutionOne(this->delta, preLayer_act, i, j, k, n);

    for (unsigned n=0; n<this->delta.n_slices; n++)
        for (unsigned j=0; j<this->delta.n_cols; j++)
            for (unsigned i=0; i<this->delta.n_rows; i++)
                this->delta_bias(n) += this->delta(i, j, n);

/*    cout << layerName << endl;
    cout << "Cumulation Conv delta_weight" << endl;
    cout << this->delta_weight << endl;
    cout << "Cumulation Conv delta_bias" << endl;
    cout << this->delta_bias << endl;
    */
}


double ConvLayer::DeltaConvolutionOne(const Cube& input, const Cube& output, unsigned& indexi, const unsigned& indexj, const unsigned& indexk, const unsigned& indexn) {

    double sum = 0.;
    for (unsigned j=0; j<input.n_cols; j++)
        for (unsigned i=0; i<input.n_rows; i++)
            sum += input(i, j, indexn) * output(i+indexi, j+indexj, indexk);

    return sum;
}



void ConvLayer::WithoutMomentum(const double& learningRate, const double& regularization, const unsigned& N_train, const unsigned& minibatchSize) {

    for (unsigned k=0; k<this->weight.size(); k++) {
        this->weight(k) *= (1. - regularization * learningRate / (double) N_train);
        this->weight(k) -= learningRate * this->delta_weight(k) / (double) minibatchSize;
    }
    bias -= learningRate * delta_bias / (double) minibatchSize;

/*    cout << layerName << endl;
    cout << "Update Conv Layer without Momentum" << endl;
    cout << "Update Conv Layer : weight" << endl;
    cout << this->weight << endl;
    cout << "Update Conv Layer : bias" << endl;
    cout << this->bias << endl;
    */
}

void ConvLayer::Momentum(const double& learningRate, const double& regularization,
        const double& momentum, const unsigned& N_train, const unsigned& minibatchSize) {

    for (unsigned k=0; k<this->weight.size(); k++) {
        this->velocity_weight(k) = momentum * this->velocity_weight(k)
                - this->weight(k) * regularization * learningRate / (double) N_train
                - learningRate * this->delta_weight(k) / (double) minibatchSize;
        this->weight(k) += this->velocity_weight(k);
    }

    this->velocity_bias = momentum * velocity_bias - learningRate * this->delta_bias / (double) minibatchSize;
    this->bias += this->velocity_bias;

/*    cout << layerName << endl;
    cout << "Update Conv Layer with Momentum" << endl;
    cout << "weight" << endl;
    cout << this->weight << endl;
    cout << "bias" << endl;
    cout << this->bias << endl;
    */
}

void ConvLayer::NesterovMomentum(const double& learningRate, const double& regularization,
        const double& momentum, const unsigned& N_train, const unsigned& minibatchSize) {

    Weights velocity_weight_prev = this->velocity_weight;
    Bias velocity_bias_prev = this->velocity_bias;

    for (unsigned k=0; k<this->weight.size(); k++) {
        this->velocity_weight(k) = momentum * this->velocity_weight(k) - this->weight(k) * regularization * learningRate / (double) N_train
            - learningRate * this->delta_weight(k) / (double) minibatchSize;
        this->weight(k) += (1. + momentum) * this->velocity_weight(k) - momentum * velocity_weight_prev(k);
    }

    this->velocity_bias = momentum * this->velocity_bias - learningRate * this->delta_bias / (double) minibatchSize;
    this->bias += (1. + momentum) * this->velocity_bias - momentum * velocity_bias_prev;

/*    cout << layerName << endl;
    cout << "Update Conv Layer with Nesterov Momentum" << endl;
    for (unsigned k=0; k<this->weight.size(); k++) {
        cout << "Update Conv Layer : weight " << endl;
        cout << this->weight(k) << endl;
    }
    cout << "Update Conv Layer : bias " << endl;
    cout << this->bias << endl;
    */
}










/***************************************************************
 * Pooling Layer
***************************************************************/
class PoolLayer : public Layer {
    public:
        PoolLayer();
        
        void Initialize_Layer(const string& layerName, const arma::ivec& input_dimension, const arma::ivec& pooling_size,
                const ActivationFuntion_Type& actFunc, const double& dropout);

        void PrintWeight() const;
        void PrintBias() const;
        void WriteResults(const string& filename) const;
        void WriteResults(const string& filename, const bool& append) const;

        void InitializeDeltaParameters();

        void Forward(const Cube& input);
        void Pooling(const Cube& input);
        double MaxPooling(const Cube& input, const unsigned& indexl, const unsigned& indexm, const unsigned& indexk);
        double L2Pooling(const Cube& input, const unsigned& indexl, const unsigned& indexm, const unsigned& indexk);


        void Backward(const Layer& nextLayer);
        void Backward(const arma::ivec& t, const CostFunction_Type& cost);

        double reverseConvolutionOne(const Cube& nextLayer_delta, const Weights nextLayer_weight,
                const unsigned& indexi, const unsigned& indexj, const unsigned& indexk, const arma::ivec filter_dimension);

        void Cumulation(const Cube& preLayer_act);

        void WithoutMomentum(const double& learningRate, const double& regularization, const unsigned& N_train, const unsigned& minibatchSize);
        void Momentum(const double& learningRate, const double& regularization,
                const double& momentum, const unsigned& N_train, const unsigned& minibatchSize);
        void NesterovMomentum(const double& learningRate, const double& regularization,
                const double& momentum, const unsigned& N_train, const unsigned& minibatchSize);


    public:
        arma::ivec pooling_size;
};

PoolLayer::PoolLayer() {};

void PoolLayer::Initialize_Layer(const string& layerName, const arma::ivec& input_dimension, const arma::ivec& pooling_size,
        const ActivationFuntion_Type& actFunc, const double& dropout) {

    this->layerName = layerName;
    this->layerType = Pool;
    this->actFunc = actFunc;
    this->dropout = 0.;

    this->input_dimension = input_dimension;
    this->pooling_size = pooling_size;

    if ( this->input_dimension.size() != 3 ) {
        cout << "Usage: Wrong input data dimension (width, height, # of input feature maps) " << endl;
        exit(1);
    }
    if ( pooling_size.size() != 2 ) {
        cout << "Usage: Wrong pooling size (width, height) " << endl;
        exit(1);
    }

    unsigned feature_maps_height = (unsigned) (this->input_dimension(0) / pooling_size(0));
    unsigned feature_maps_width = (unsigned) (this->input_dimension(1) / pooling_size(1));
    this->output_dimension.set_size(3);
    this->output_dimension(0) = feature_maps_height;
    this->output_dimension(1) = feature_maps_width;
    this->output_dimension(2) = this->input_dimension(2);
    this->activation.set_size(feature_maps_height, feature_maps_width, this->input_dimension(2));
    this->delta.set_size(feature_maps_height, feature_maps_width, this->input_dimension(2));
}


void PoolLayer::PrintWeight() const {
    cout << "################################" << endl;
    cout << layerName << endl;
    cout << "################################" << endl;
    cout << "No weight " << endl << endl;
}

void PoolLayer::PrintBias() const {
    cout << "No Bias " << endl << endl;
}


void PoolLayer::WriteResults(const string& filename) const {
    ofstream fsave(filename.c_str());

    fsave << "################################" << endl;
    fsave << layerName << endl;
    fsave << "################################" << endl;
    fsave << "No weight " << endl << endl;
    fsave << "No Bias " << endl << endl;
}

void PoolLayer::WriteResults(const string& filename, const bool& append) const {
    ofstream fsave;
    if ( !append )
        fsave.open(filename.c_str());
    else
        fsave.open(filename.c_str(), fstream::out | fstream::app);

    fsave << "################################" << endl;
    fsave << layerName << endl;
    fsave << "################################" << endl;
    fsave << "No weight " << endl << endl;
    fsave << "No Bias " << endl << endl;
}


void PoolLayer::InitializeDeltaParameters() {}

void PoolLayer::Forward(const Cube& input) {

    Pooling(input);
/*    cout << layerName << endl;
    cout << "Forward Pool input" << endl;
    cout << input << endl;
    cout << "Forward Pool activation" << endl;
    cout << activation << endl;
    */
}


void PoolLayer::Pooling(const Cube& input) {

    for (unsigned k=0; k<this->output_dimension(2); k++) {
        for (unsigned m=0; m<this->output_dimension(1); m++)
            for (unsigned l=0; l<this->output_dimension(0); l++)
                this->activation(l, m, k) = MaxPooling(input, l, m, k);
//                this->activation(l, m, k) = L2Pooling(input, l, m, k);
    }
}

double PoolLayer::MaxPooling(const Cube& input, const unsigned& indexl, const unsigned& indexm, const unsigned& indexk) {

    double max = -100000000.;
    for (unsigned j=0; j<pooling_size(1); j++)
        for (unsigned i=0; i<pooling_size(0); i++)
            if ( max < input(indexl * pooling_size(0) + i, indexm * pooling_size(1) + j, indexk) )
                max = input(indexl * pooling_size(0) + i, indexm * pooling_size(1) + j, indexk);

    return max;
}

double PoolLayer::L2Pooling(const Cube& input, const unsigned& indexl, const unsigned& indexm, const unsigned& indexk) {

    double sum = 0.;
    for (unsigned j=0; j<pooling_size(1); j++)
        for (unsigned i=0; i<pooling_size(0); i++)
            sum += pow(input(indexl * pooling_size(0) + i, indexm * pooling_size(1) + j, indexk), 2);

    return sqrt(sum / arma::prod(pooling_size));
}



void PoolLayer::Backward(const Layer& nextLayer) {

    if ( nextLayer.layerType == FC  ||  nextLayer.layerType == Soft ) {

        Vector nextLayer_delta = ReshapeCubetoVector(nextLayer.delta);
        this->delta = ReshapeVectortoCube((nextLayer_delta.t() * nextLayer.weight(0).slice(0)).t(),
            this->output_dimension(0), this->output_dimension(1), this->output_dimension(2));
    }
    else if ( nextLayer.layerType == Conv ) {

        arma::ivec filter_dimension(3);
        filter_dimension(0) = this->output_dimension(0) - nextLayer.output_dimension(0) + 1;
        filter_dimension(1) = this->output_dimension(1) - nextLayer.output_dimension(1) + 1;
        filter_dimension(2) = nextLayer.output_dimension(2);

        for (unsigned k=0; k<this->output_dimension(2); k++)
            for (unsigned j=0; j<this->output_dimension(1); j++)
                for (unsigned i=0; i<this->output_dimension(0); i++)
                    this->delta(i, j, k) = reverseConvolutionOne(nextLayer.delta, nextLayer.weight,
                            i, j, k, filter_dimension);
    }
    else {
        cout << "Usage: Not yet implemented in PoolLayer : next Pool" << endl;
        exit(1);
    }

/*    cout << layerName << endl;
    cout << "Backward Pool nextLayer delta" << endl;
    cout << nextLayer_delta << endl;
    cout << "Backward Pool nextLayer weight" << endl;
    cout << nextLayer.weight << endl;
    cout << "Backward Pool delta" << endl;
    cout << this->delta << endl;
    */
}

double PoolLayer::reverseConvolutionOne(const Cube& nextLayer_delta, const Weights nextLayer_weight,
        const unsigned& indexi, const unsigned& indexj, const unsigned& indexk, const arma::ivec filter_dimension) {

    double sum = 0.;
    for (unsigned n=0; n<filter_dimension(2); n++)
        for (unsigned m=0; m<filter_dimension(1); m++)
            for (unsigned l=0; l<filter_dimension(0); l++)
                if ( indexi-l >=0  &&  indexi-l < nextLayer_delta.n_rows
                    &&  indexj-m >=0  &&  indexj-m < nextLayer_delta.n_cols )
                    sum += nextLayer_weight(n)(l, m, indexk) * nextLayer_delta(indexi-l, indexj-m, indexk);

    return sum;
}


void PoolLayer::Backward(const arma::ivec& t, const CostFunction_Type& cost) {
    cout << "Usage: No Backward function in PoolLayer" << endl;
    exit(1);
}


void PoolLayer::Cumulation(const Cube& preLayer_act) {
//    cout << "Cumulation Pool" << endl;
}

void PoolLayer::WithoutMomentum(const double& learningRate, const double& regularization, const unsigned& N_train, const unsigned& minibatchSize) {
//    cout << "Update Pool Layer without Momentum" << endl;
}
void PoolLayer::Momentum(const double& learningRate, const double& regularization,
        const double& momentum, const unsigned& N_train, const unsigned& minibatchSize) {
//    cout << "Update Pool Layer with Momentum" << endl;
}
void PoolLayer::NesterovMomentum(const double& learningRate, const double& regularization,
        const double& momentum, const unsigned& N_train, const unsigned& minibatchSize) {
//    cout << "Update Pool Layer with Nesterov Momentum" << endl;
}






/***************************************************************
 * Fully Connected Layer
***************************************************************/
class FullyConnectedLayer : public Layer {
    public:
        FullyConnectedLayer();
        
        void Initialize_Layer(const string& layerName, const arma::ivec& input_dimension, const arma::ivec& output_dimension,
                const ActivationFuntion_Type& actFunc, const double& dropout);

        void PrintWeight() const;
        void PrintBias() const;
        void WriteResults(const string& filename) const;
        void WriteResults(const string& filename, const bool& append) const;

        void InitializeDeltaParameters();

        void Forward(const Cube& input_cube);
        void Activation();

        
        void Backward(const Layer& nextLayer);
        void Backward(const arma::ivec& t, const CostFunction_Type& cost);

        void Cumulation(const Cube& preLayer_act);

        void WithoutMomentum(const double& learningRate, const double& regularization, const unsigned& N_train, const unsigned& minibatchSize);
        void Momentum(const double& learningRate, const double& regularization,
                const double& momentum, const unsigned& N_train, const unsigned& minibatchSize);
        void NesterovMomentum(const double& learningRate, const double& regularization,
                const double& momentum, const unsigned& N_train, const unsigned& minibatchSize);


    public:
        Bias bias;
        Bias delta_bias;
        Bias velocity_bias;
};

FullyConnectedLayer::FullyConnectedLayer() {};

void FullyConnectedLayer::Initialize_Layer(const string& layerName, const arma::ivec& input_dimension, const arma::ivec& output_dimension,
        const ActivationFuntion_Type& actFunc, const double& dropout) {

    this->layerName = layerName;
    this->layerType = FC;
    this->actFunc = actFunc;
    if ( dropout >= 1.  ||  dropout < 0. ) cout << "Wrong dropout percent " << endl;
    this->dropout = dropout;

    this->input_dimension = input_dimension;
    this->output_dimension = output_dimension;
    unsigned n_input = arma::prod(this->input_dimension);
    unsigned n_output = arma::prod(this->output_dimension);


    this->weight.set_size(1);
    this->weight(0).set_size(n_output, n_input, 1);
    bias.set_size(n_output);

    this->summation.set_size(n_output, 1, 1);
    this->activation.set_size(n_output, 1, 1);

    this->delta.set_size(n_output, 1, 1);

    this->delta_weight.set_size(1);
    this->delta_weight(0).set_size(n_output, n_input, 1);
    delta_bias.set_size(n_output);

    this->velocity_weight.set_size(1);
    this->velocity_weight(0).set_size(n_output, n_input, 1);
    this->velocity_bias.set_size(n_output);


    //  weight Initialize from Gaussian distribution
    double std_dev = 1. / sqrt(n_input);
    boost::random::normal_distribution<> normal_dist(0., std_dev);          //  Choose a distribution
    boost::random::variate_generator<boost::random::mt19937 &,
        boost::random::normal_distribution<> > nrnd(rng, normal_dist);      //  link the Generator to the distribution

    for (unsigned j=0; j<this->weight(0).n_cols; j++)
        for (unsigned i=0; i<this->weight(0).n_rows; i++)
            this->weight(0)(i, j, 0) = nrnd();

    boost::random::normal_distribution<> normal_dist1(0., 1.);              //  Choose a distribution
    boost::random::variate_generator<boost::random::mt19937 &,
        boost::random::normal_distribution<> > nrnd1(rng, normal_dist1);    //  link the Generator to the distribution

    for (unsigned n=0; n<this->bias.size(); n++)
        this->bias(n) = nrnd1();

    this->velocity_weight(0).zeros();
    this->velocity_bias.zeros();
}


void FullyConnectedLayer::PrintWeight() const {
    cout << "################################" << endl;
    cout << layerName << endl;
    cout << "################################" << endl;
    this->weight(0).raw_print("fully connected layer weight ");
    cout << endl;
}

void FullyConnectedLayer::PrintBias() const {
    this->bias.raw_print("fully connected layer bias ");
    cout << endl;
}

void FullyConnectedLayer::WriteResults(const string& filename) const {
    ofstream fsave(filename.c_str());
    fsave.precision(10);
    fsave.setf(ios::fixed);

    fsave << "################################" << endl;
    fsave << layerName << endl;
    fsave << "################################" << endl;
    this->weight(0).raw_print(fsave, "fully connected layer weight ");
    fsave << endl;
    this->bias.raw_print(fsave, "fully connected layer bias ");
    fsave << endl;
}

void FullyConnectedLayer::WriteResults(const string& filename, const bool& append) const {
    ofstream fsave;
    if ( !append )
        fsave.open(filename.c_str());
    else
        fsave.open(filename.c_str(), fstream::out | fstream::app);
    fsave.precision(10);
    fsave.setf(ios::fixed);

    fsave << "################################" << endl;
    fsave << layerName << endl;
    fsave << "################################" << endl;
    this->weight(0).raw_print(fsave, "fully connected layer weight ");
    fsave << endl;
    this->bias.raw_print(fsave, "fully connected layer bias ");
    fsave << endl;
}


void FullyConnectedLayer::InitializeDeltaParameters() {

    this->delta_weight(0).zeros();
    this->delta_bias.zeros();
}


void FullyConnectedLayer::Forward(const Cube& input_cube) {

    Vector input = ReshapeCubetoVector(input_cube);
    this->summation = ReshapeVectortoCube(this->weight(0).slice(0) * input + bias, this->output_dimension(0), this->output_dimension(1), this->output_dimension(2));
    Activation();

/*    cout << layerName << endl;
    cout << "Forward FC input" << endl;
    cout << input_cube << endl;
    cout << "Forward FC summation" << endl;
    cout << summation << endl;
    cout << "Forward FC activation" << endl;
    cout << activation << endl;
    */

    if ( this->dropout != 0 ) {
        boost::random::uniform_real_distribution<> uniform_real_dist(0, 1);      //  Choose a distribution
        boost::random::variate_generator<boost::mt19937 &,
            boost::random::uniform_real_distribution<> > urnd(rng, uniform_real_dist);      //  link the Generator to the distribution

        for (unsigned i=0; i<this->activation.n_rows; i++) {
            if ( urnd() < this->dropout ) this->activation(i, 0, 0) = 0.;
            else this->activation(i, 0, 0) *= 1./(1. - this->dropout);
        }
    }
}

void FullyConnectedLayer::Activation() {
    
    if ( actFunc == Sigmoid )
        this->activation = 1. / (1. + exp(-this->summation));
    else if ( actFunc == Tanh )
        this->activation = tanh(this->summation);
    else if ( actFunc == ReLU )
        for (unsigned i=0; i<this->summation.n_rows; i++)
            this->activation(i, 0, 0) = this->summation(i, 0, 0) > 0 ? this->summation(i, 0, 0) : 0.;
}




void FullyConnectedLayer::Backward(const Layer& nextLayer) {

    if ( nextLayer.layerType != FC  &&  nextLayer.layerType != Soft ) {
        cout << "Usage: nextLayer of FCLayer must be FC or Soft" << endl;
        exit(1);
    }

    Vector nextLayer_delta = ReshapeCubetoVector(nextLayer.delta);
    if ( actFunc == Sigmoid )
        this->delta = ReshapeVectortoCube((nextLayer_delta.t() * nextLayer.weight(0).slice(0)).t(),
            this->output_dimension(0), this->output_dimension(1), this->output_dimension(2)) % DerivativeSigmoid();
    else if ( actFunc == Tanh )
        this->delta = ReshapeVectortoCube((nextLayer_delta.t() * nextLayer.weight(0).slice(0)).t(),
            this->output_dimension(0), this->output_dimension(1), this->output_dimension(2)) % DerivativeTanh();
    else if ( actFunc == ReLU )
        this->delta = ReshapeVectortoCube((nextLayer_delta.t() * nextLayer.weight(0).slice(0)).t(),
            this->output_dimension(0), this->output_dimension(1), this->output_dimension(2)) % DerivativeReLU();

/*    cout << layerName << endl;
    cout << "Backward FC nextLayer delta" << endl;
    cout << nextLayer_delta << endl;
    cout << "Backward FC nextLayer weight" << endl;
    cout << nextLayer.weight << endl;
    cout << "Backward FC delta" << endl;
    cout << this->delta << endl;


    cout << "Backward FC *********" << endl;
    cout << (nextLayer_delta.t() * nextLayer.weight(0).slice(0)).t() << endl;
    cout << "Backward FC *********" << endl;
    cout << DerivativeSigmoid() << endl;
    */
}

void FullyConnectedLayer::Backward(const arma::ivec& t, const CostFunction_Type& cost) {

    Cube target = ReshapeVectortoCube(t, this->output_dimension(0), this->output_dimension(1), this->output_dimension(2));
    if ( cost != CrossEntropy ) {
        if ( actFunc == Sigmoid )
            this->delta = (this->activation - target) % DerivativeSigmoid();
        else if ( actFunc == Tanh )
            this->delta = (this->activation - target) % DerivativeTanh();
        else if ( actFunc == ReLU )
            this->delta = (this->activation - target) % DerivativeReLU();
    }
    else {
        this->delta = (this->activation - target);
    }

/*    cout << layerName << endl;
    cout << "Backward FC target" << endl;
    cout << target << endl;
    cout << "Backward FC delta" << endl;
    cout << this->delta << endl;
    */
}


void FullyConnectedLayer::Cumulation(const Cube& preLayer_act) {

    Vector x = ReshapeCubetoVector(preLayer_act);
    this->delta_weight(0).slice(0) += this->delta.slice(0) * x.t();
    this->delta_bias += this->delta.slice(0);

/*    cout << layerName << endl;
    cout << "Cumulation FC delta_weight" << endl;
    cout << this->delta_weight(0) << endl;
    cout << "Cumulation FC delta_bias" << endl;
    cout << this->delta_bias << endl;
    */
}



void FullyConnectedLayer::WithoutMomentum(const double& learningRate, const double& regularization, const unsigned& N_train, const unsigned& minibatchSize) {

    this->weight(0) *= (1. - regularization * learningRate / (double) N_train);
    this->weight(0) -= learningRate * this->delta_weight(0) / (double) minibatchSize;
    this->bias -= learningRate * this->delta_bias / (double) minibatchSize;

/*    cout << layerName << endl;
    cout << "Update FC Layer without Momentum" << endl;
    cout << "Update FC Layer : weight " << endl;
    cout << this->weight(0) << endl;
    cout << "Update FC Layer : bias " << endl;
    cout << this->bias << endl;
    */
}

void FullyConnectedLayer::Momentum(const double& learningRate, const double& regularization,
        const double& momentum, const unsigned& N_train, const unsigned& minibatchSize) {

    this->velocity_weight(0) = momentum * this->velocity_weight(0)
            - this->weight(0) * regularization * learningRate / (double) N_train
            - learningRate * this->delta_weight(0) / (double) minibatchSize;
    this->weight(0) += this->velocity_weight(0);

    this->velocity_bias = momentum * this->velocity_bias - learningRate * this->delta_bias / (double) minibatchSize;
    this->bias += this->velocity_bias;

/*    cout << layerName << endl;
    cout << "Update FC Layer with Momentum" << endl;
    cout << "Update FC Layer : weight " << endl;
    cout << this->weight(0) << endl;
    cout << "Update FC Layer : bias " << endl;
    cout << this->bias << endl;
    */
}

void FullyConnectedLayer::NesterovMomentum(const double& learningRate, const double& regularization,
        const double& momentum, const unsigned& N_train, const unsigned& minibatchSize) {

    Weights velocity_weight_prev = this->velocity_weight;
    Bias velocity_bias_prev = this->velocity_bias;

    this->velocity_weight(0) = momentum * this->velocity_weight(0) - this->weight(0) * regularization * learningRate / (double) N_train
        - learningRate * this->delta_weight(0) / (double) minibatchSize;
    this->weight(0) += (1. + momentum) * this->velocity_weight(0) - momentum * velocity_weight_prev(0);

    this->velocity_bias = momentum * this->velocity_bias - learningRate * this->delta_bias / (double) minibatchSize;
    this->bias += (1. + momentum) * this->velocity_bias - momentum * velocity_bias_prev;

/*    cout << layerName << endl;
    cout << "Update FC Layer with Nesterov Momentum" << endl;
    cout << "Update FC Layer : weight " << endl;
    cout << this->weight(0) << endl;
    cout << "Update FC Layer : bias " << endl;
    cout << this->bias << endl;
    */
}












/***************************************************************
 * Softmax Layer
***************************************************************/
class SoftmaxLayer : public Layer {
    public:
        SoftmaxLayer();
        
        void Initialize_Layer(const string& layerName, const arma::ivec& input_dimension, const arma::ivec& output_dimension,
                const ActivationFuntion_Type& actFunc, const double& dropout);

        void PrintWeight() const;
        void PrintBias() const;
        void WriteResults(const string& filename) const;
        void WriteResults(const string& filename, const bool& append) const;

        void InitializeDeltaParameters();

        void Forward(const Cube& input_cube);
        void Activation();


        void Backward(const Layer& nextLayer);
        void Backward(const arma::ivec& t, const CostFunction_Type& cost);
        Cube DerivativeSoftmax();

        void Cumulation(const Cube& preLayer_act);

        void WithoutMomentum(const double& learningRate, const double& regularization, const unsigned& N_train, const unsigned& minibatchSize);
        void Momentum(const double& learningRate, const double& regularization,
                const double& momentum, const unsigned& N_train, const unsigned& minibatchSize);
        void NesterovMomentum(const double& learningRate, const double& regularization,
                const double& momentum, const unsigned& N_train, const unsigned& minibatchSize);


    public:
};

SoftmaxLayer::SoftmaxLayer() {};

void SoftmaxLayer::Initialize_Layer(const string& layerName, const arma::ivec& input_dimension, const arma::ivec& output_dimension,
        const ActivationFuntion_Type& actFunc, const double& dropout) {

    this->layerName = layerName;
    this->layerType = Soft;
    this->actFunc = actFunc;
    if ( dropout >= 1.  ||  dropout < 0. ) cout << "Wrong dropout percent " << endl;
    this->dropout = dropout;

    this->input_dimension = input_dimension;
    this->output_dimension = output_dimension;
    unsigned n_input = arma::prod(this->input_dimension);
    unsigned n_output = arma::prod(this->output_dimension);


    this->weight.set_size(1);
    this->weight(0).set_size(n_output, n_input, 1);
    this->bias.set_size(n_output);

    this->summation.set_size(n_output, 1, 1);
    this->activation.set_size(n_output, 1, 1);

    this->delta.set_size(n_output, 1, 1);
    this->delta_weight.set_size(1);
    this->delta_weight(0).set_size(n_output, n_input, 1);
    this->delta_bias.set_size(n_output);

    this->velocity_weight.set_size(1);
    this->velocity_weight(0).set_size(n_output, n_input, 1);
    this->velocity_bias.set_size(n_output);


    //  weight Initialize from Gaussian distribution
    double std_dev = 1. / sqrt(n_input);
    boost::random::normal_distribution<> normal_dist(0., std_dev);         //  Choose a distribution
    boost::random::variate_generator<boost::random::mt19937 &,
        boost::random::normal_distribution<> > nrnd(rng, normal_dist);      //  link the Generator to the distribution

    for (unsigned j=0; j<this->weight(0).n_cols; j++)
        for (unsigned i=0; i<this->weight(0).n_rows; i++)
            this->weight(0)(i, j, 0) = nrnd();

    boost::random::normal_distribution<> normal_dist1(0., 1.);         //  Choose a distribution
    boost::random::variate_generator<boost::random::mt19937 &,
        boost::random::normal_distribution<> > nrnd1(rng, normal_dist1);      //  link the Generator to the distribution

    for (unsigned n=0; n<this->bias.size(); n++)
        this->bias(n) = nrnd1();

    this->velocity_weight(0).zeros();
    this->velocity_bias.zeros();
}


void SoftmaxLayer::PrintWeight() const {
    cout << "################################" << endl;
    cout << layerName << endl;
    cout << "################################" << endl;
    this->weight(0).raw_print("softmax layer weight ");
    cout << endl;
}

void SoftmaxLayer::PrintBias() const {
    this->bias.raw_print("softmax layer bias ");
    cout << endl;
}

void SoftmaxLayer::WriteResults(const string& filename) const {
    ofstream fsave(filename.c_str());
    fsave.precision(10);
    fsave.setf(ios::fixed);

    fsave << "################################" << endl;
    fsave << layerName << endl;
    fsave << "################################" << endl;
    this->weight(0).raw_print(fsave, "softmax layer weight ");
    fsave << endl;
    this->bias.raw_print(fsave, "softmax layer bias ");
    fsave << endl;
}

void SoftmaxLayer::WriteResults(const string& filename, const bool& append) const {
    ofstream fsave;
    if ( !append )
        fsave.open(filename.c_str());
    else
        fsave.open(filename.c_str(), fstream::out | fstream::app);
    fsave.precision(10);
    fsave.setf(ios::fixed);

    fsave << "################################" << endl;
    fsave << layerName << endl;
    fsave << "################################" << endl;
    this->weight(0).raw_print(fsave, "softmax layer weight ");
    fsave << endl;
    this->bias.raw_print(fsave, "softmax layer bias ");
    fsave << endl;
}


void SoftmaxLayer::InitializeDeltaParameters() {

    this->delta_weight(0).zeros();
    this->delta_bias.zeros();
}


void SoftmaxLayer::Forward(const Cube& input_cube) {

    Vector input = ReshapeCubetoVector(input_cube);
    this->summation = ReshapeVectortoCube(this->weight(0).slice(0) * input + bias, this->output_dimension(0), this->output_dimension(1), this->output_dimension(2));
    Activation();

/*    cout << layerName << endl;
    cout << "Forward Soft input" << endl;
    cout << input_cube << endl;
    cout << "Forward Soft summation" << endl;
    cout << summation << endl;
    cout << "Forward Soft activation" << endl;
    cout << activation << endl;
    */

    if ( this->dropout != 0 ) {
        boost::random::uniform_real_distribution<> uniform_real_dist(0, 1);      //  Choose a distribution
        boost::random::variate_generator<boost::mt19937 &,
            boost::random::uniform_real_distribution<> > urnd(rng, uniform_real_dist);      //  link the Generator to the distribution

        for (unsigned i=0; i<this->activation.n_rows; i++) {
            if ( urnd() < this->dropout ) this->activation(i, 0, 0) = 0.;
            else this->activation(i, 0, 0) *= 1./(1. - this->dropout);
        }
    }
}



void SoftmaxLayer::Activation() {
    this->activation = exp(this->summation) / arma::accu(exp(this->summation));
}




void SoftmaxLayer::Backward(const Layer& nextLayer) {
    cout << "Usage: No Backward function in SoftmaxLayer" << endl;
    exit(1);
}

void SoftmaxLayer::Backward(const arma::ivec& t, const CostFunction_Type& cost) {

    Cube target = ReshapeVectortoCube(t, this->output_dimension(0), this->output_dimension(1), this->output_dimension(2));
    if ( cost != CrossEntropy )
        this->delta = (this->activation - target) % DerivativeSoftmax();
    else
        this->delta = (this->activation - target);

/*    cout << layerName << endl;
    cout << "Backward Soft target" << endl;
    cout << target << endl;
    cout << "Backward Soft delta" << endl;
    cout << this->delta << endl;
    */
}

Cube SoftmaxLayer::DerivativeSoftmax() {
    return this->activation % (1. - this->activation);
}


void SoftmaxLayer::Cumulation(const Cube& preLayer_act) {

    Vector x = ReshapeCubetoVector(preLayer_act);
    this->delta_weight(0).slice(0) += this->delta.slice(0) * x.t();
    this->delta_bias += this->delta.slice(0);

/*    cout << layerName << endl;
    cout << "Cumulation Soft delta_weight" << endl;
    cout << this->delta_weight(0) << endl;
    cout << "Cumulation Soft delta_bias" << endl;
    cout << this->delta_bias << endl;
    */
}




void SoftmaxLayer::WithoutMomentum(const double& learningRate, const double& regularization, const unsigned& N_train, const unsigned& minibatchSize) {

    this->weight(0) *= (1. - regularization * learningRate / (double) N_train);
    this->weight(0) -= learningRate * this->delta_weight(0) / (double) minibatchSize;
    this->bias -= learningRate * this->delta_bias / (double) minibatchSize;

/*    cout << layerName << endl;
    cout << "Update Soft Layer without Momentum" << endl;
    cout << "Update Soft Layer : weight " << endl;
    cout << this->weight(0) << endl;
    cout << "Update Soft Layer : bias " << endl;
    cout << this->bias << endl;
    */
}

void SoftmaxLayer::Momentum(const double& learningRate, const double& regularization,
        const double& momentum, const unsigned& N_train, const unsigned& minibatchSize) {

    this->velocity_weight(0) = momentum * this->velocity_weight(0)
            - this->weight(0) * regularization * learningRate / (double) N_train
            - learningRate * this->delta_weight(0) / (double) minibatchSize;
    this->weight(0) += this->velocity_weight(0);

    this->velocity_bias = momentum * this->velocity_bias - learningRate * this->delta_bias / (double) minibatchSize;
    this->bias += this->velocity_bias;

/*    cout << layerName << endl;
    cout << "Update Soft Layer with Momentum" << endl;
    cout << "Update Soft Layer : weight " << endl;
    cout << this->weight(0) << endl;
    cout << "Update Soft Layer : bias " << endl;
    cout << this->bias << endl;
    */
}

void SoftmaxLayer::NesterovMomentum(const double& learningRate, const double& regularization,
        const double& momentum, const unsigned& N_train, const unsigned& minibatchSize) {

    Weights velocity_weight_prev = this->velocity_weight;
    Bias velocity_bias_prev = this->velocity_bias;

    this->velocity_weight(0) = momentum * this->velocity_weight(0) - this->weight(0) * regularization * learningRate / (double) N_train
        - learningRate * this->delta_weight(0) / (double) minibatchSize;
    this->weight(0) += (1. + momentum) * this->velocity_weight(0) - momentum * velocity_weight_prev(0);

    this->velocity_bias = momentum * this->velocity_bias - learningRate * this->delta_bias / (double) minibatchSize;
    this->bias += (1. + momentum) * this->velocity_bias - momentum * velocity_bias_prev;

/*    cout << layerName << endl;
    cout << "Update Soft Layer with Nesterov Momentum" << endl;
    cout << "Update Soft Layer : weight " << endl;
    cout << this->weight(0) << endl;
    cout << "Update Soft Layer : bias " << endl;
    cout << this->bias << endl;
    */
}







}

#endif

