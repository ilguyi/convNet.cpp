/***********************************************************
 * Convolutional Neural networks for multi-class classification
 *
 * 2015. 08. 25.
 * modified 2015. 09. 03.
 * by Il Gu Yi
***********************************************************/

#ifndef CONV_NEURAL_NETWORK_H
#define CONV_NEURAL_NETWORK_H

#include <boost/random.hpp>
#include <armadillo>
#include <string>
#include "convdef.h"
#include "convlayer.h"
using namespace std;
using namespace df;
using namespace convdef;
using namespace convlayer;


namespace conv {


typedef struct NeuralNetworkParameters {
    NeuralNetworkParameters() :
        N_train(0), dimension(0),
        N_valid(0), N_test(0),
        n_class(0),
        n_layers(0),
        learningRate(0.5),
        cost(CrossEntropy),
        regularization(0.0),
        momentum(0.0),
        minibatchSize(1),
        maxEpoch(100) {
        };


    unsigned N_train;
    unsigned dimension;
    unsigned N_valid;
    unsigned N_test;
    unsigned n_class;
    unsigned n_layers;
    double learningRate;
    CostFunction_Type cost;
    double regularization;
    double momentum;
    unsigned minibatchSize;
    unsigned maxEpoch;
} NNParameters;




class NeuralNetworks {
    public:
        NeuralNetworks();
        NeuralNetworks(const unsigned& N_train, const unsigned& dimension, const unsigned& N_valid, const unsigned& N_test,
            const unsigned& n_class, const unsigned& n_layers, const double& learningRate, const CostFunction_Type& cost,
            const double& regularization, const double& momentum, const unsigned& minibatchSize, const unsigned& maxEpoch);

        void ReadParameters(const string& filename);
        void ParametersSetting(const unsigned& N_train, const unsigned& dimension, const unsigned& N_valid, const unsigned& N_test,
            const unsigned& n_class, const unsigned& n_layers, const double& learningRate, const CostFunction_Type& cost,
            const double& regularization, const double& momentum, const unsigned& minibatchSize, const unsigned& maxEpoch);
        void PrintParameters() const;
        void WriteParameters(const string& filename) const;

        unsigned GetN_train() const;
        unsigned GetDimension() const;
        unsigned GetN_valid() const;
        unsigned GetN_test() const;
        unsigned GetN_class() const;

        void Initialize();
//        void Initialize(const string& initialize_type, const Weights& weight_init, const Biases& bias_init);

    public:
        void PrintWeights() const;
        void PrintBiases() const;
        void PrintResults() const;
        void WriteResults(const string& filename) const;


    private:
        void NamingFile(string& filename);
        void NamingFileStep(string& filename, const unsigned& step);


    public:
        template<typename dataType>
        void Training(df::DataFrame<dataType>& data, const unsigned& step);
        template<typename dataType>
        void Training(df::DataFrame<dataType>& data, df::DataFrame<dataType>& valid, unsigned& step);

    private:
        template<typename dataType>
        void TrainingOneStep(df::DataFrame<dataType>& data);
        template<typename dataType>
        void TrainingOneStep(df::DataFrame<dataType>& data, df::DataFrame<dataType>& valid);

        void MiniBathces(arma::field<Vector>& minibatch);
        void InitializeDeltaParameters();
        template<typename dataType>
        void TrainingMiniBatch(df::DataFrame<dataType>& data, const Vector& minibatch, double& error);

        Cube ReshapeVectortoCube(const Vector& input_vec, const arma::ivec& input_dim);
        Cube ReshapeVectortoCube(const Vector& input_vec, const unsigned& height, const unsigned& width, const unsigned& depth);
        Vector ReshapeCubetoVector(const Cube& input_cube);

        template<typename dataType>
        void FeedForward(const arma::Row<dataType>& x);

        void CostFunction(double& error, const arma::ivec& t);

        template<typename dataType>
        void Validation(df::DataFrame<dataType>& valid, double& error, double& accuracy);

        template<typename dataType>
        void BackPropagation(const arma::Row<dataType>& x, const arma::ivec& t);

        void UpdateParameter(const unsigned& minibatchSize);
        
        void WriteError(const string& filename, const double& error);
        void WriteError(const string& filename, const double& error, const double& valid_error, const double& valid_accuracy);


    public:
        template<typename dataType>
        void Test(df::DataFrame<dataType>& data, arma::uvec& predict, unsigned& step);


    private:
        Layer** layers;

        NNParameters nnParas;
};

NeuralNetworks::NeuralNetworks() {};
NeuralNetworks::NeuralNetworks(const unsigned& N_train, const unsigned& dimension, const unsigned& N_valid, const unsigned& N_test,
    const unsigned& n_class, const unsigned& n_layers, const double& learningRate, const CostFunction_Type& cost, 
    const double& regularization, const double& momentum, const unsigned& minibatchSize, const unsigned& maxEpoch) {

    nnParas.N_train = N_train;
    nnParas.dimension = dimension;
    nnParas.N_valid = N_valid;
    nnParas.N_test = N_test;
    nnParas.n_class = n_class;
    nnParas.n_layers = n_layers;
    nnParas.learningRate = learningRate;
    nnParas.cost = cost;
    nnParas.regularization = regularization;
    nnParas.momentum = momentum;
    nnParas.minibatchSize = minibatchSize;
    nnParas.maxEpoch = maxEpoch;
}



void NeuralNetworks::ReadParameters(const string& filename) {

    ifstream fin(filename.c_str());
    string s;
    for (unsigned i=0; i<4; i++) getline(fin, s);
    nnParas.N_train = stoi(s);

    getline(fin, s);    getline(fin, s);
    nnParas.dimension = stoi(s);

    getline(fin, s);    getline(fin, s);
    nnParas.N_valid = stoi(s);

    getline(fin, s);    getline(fin, s);
    nnParas.N_test = stoi(s);

    getline(fin, s);    getline(fin, s);
    nnParas.n_class = stoi(s);

    getline(fin, s);    getline(fin, s);
    nnParas.n_layers = stoi(s);

    getline(fin, s);    getline(fin, s);
    nnParas.learningRate = stod(s);

    getline(fin, s);    getline(fin, s);    getline(fin, s);
    getline(fin, s);    getline(fin, s);
    if ( s == "CrossEntropy" ) nnParas.cost = CrossEntropy;
    else if ( s == "Quadratic" ) nnParas.cost = Quadratic;
    else {
        cout << "Usage: Wrong Cost function type" << endl;
        cout << "you must type CrossEntropy or Quadratic" << endl;
    }

    getline(fin, s);    getline(fin, s);
    nnParas.regularization = stod(s);
    
    getline(fin, s);    getline(fin, s);
    nnParas.momentum = stod(s);

    getline(fin, s);    getline(fin, s);
    nnParas.minibatchSize = stoi(s);

    getline(fin, s);    getline(fin, s);
    nnParas.maxEpoch = stoi(s);
}



void NeuralNetworks::ParametersSetting(const unsigned& N_train, const unsigned& dimension, const unsigned& N_valid, const unsigned& N_test,
    const unsigned& n_class, const unsigned& n_layers, const double& learningRate, const CostFunction_Type& cost, 
    const double& regularization, const double& momentum, const unsigned& minibatchSize, const unsigned& maxEpoch) {

    nnParas.N_train = N_train;
    nnParas.dimension = dimension;
    nnParas.N_valid = N_valid;
    nnParas.N_test = N_test;
    nnParas.n_class = n_class;
    nnParas.n_layers = n_layers;
    nnParas.learningRate = learningRate;
    nnParas.cost = cost;
    nnParas.regularization = regularization;
    nnParas.momentum = momentum;
    nnParas.minibatchSize = minibatchSize;
    nnParas.maxEpoch = maxEpoch;
}

void NeuralNetworks::PrintParameters() const {
    cout << "##################################"    << endl;
    cout << "##  Neural Networks Parameters  ##"    << endl;
    cout << "##################################"    << endl << endl;
    cout << "Number of train data: "                << nnParas.N_train << endl;
    cout << "dimension: "                           << nnParas.dimension << endl;
    cout << "Number of validation data: "           << nnParas.N_valid << endl;
    cout << "Number of test data: "                 << nnParas.N_test << endl;
    cout << "number of class: "                     << nnParas.n_class << endl ;
    cout << "number of layers: "                    << nnParas.n_layers << endl ;
    cout << "learning rate: "                       << nnParas.learningRate << endl;

    if ( nnParas.cost == CrossEntropy ) cout << "cost function: Cross Entropy" << endl;
    else cout << "cost function: Quadratic" << endl;

    cout << "regularization rate: "                 << nnParas.regularization << endl;
    cout << "momentum rate: "                       << nnParas.momentum << endl;
    cout << "minibatch size: "                      << nnParas.minibatchSize << endl;
    cout << "iteration max epochs: "                << nnParas.maxEpoch << endl << endl;
}

void NeuralNetworks::WriteParameters(const string& filename) const {
    ofstream fsave(filename.c_str());
    fsave << "##################################"   << endl;
    fsave << "##  Neural Networks Parameters  ##"   << endl;
    fsave << "##################################"   << endl << endl;
    fsave << "Number of train data: "               << nnParas.N_train << endl;
    fsave << "dimension: "                          << nnParas.dimension << endl;
    fsave << "Number of validation data: "          << nnParas.N_valid << endl;
    fsave << "Number of test data: "                << nnParas.N_test << endl;
    fsave << "number of class: "                    << nnParas.n_class << endl ;
    fsave << "number of layers: "                   << nnParas.n_layers << endl ;
    fsave << "learning_Rate: "                      << nnParas.learningRate << endl;

    if ( nnParas.cost == CrossEntropy ) fsave << "cost function: Cross Entropy" << endl;
    else fsave << "cost function: Quadratic" << endl;

    fsave << "regularization rate: "                << nnParas.regularization << endl;
    fsave << "momentum rate: "                      << nnParas.momentum << endl;
    fsave << "minibatch size: "                     << nnParas.minibatchSize << endl;
    fsave << "iteration max epochs: "               << nnParas.maxEpoch << endl << endl;
    fsave.close();
}


unsigned NeuralNetworks::GetN_train() const { return nnParas.N_train; }
unsigned NeuralNetworks::GetDimension() const { return nnParas.dimension; }
unsigned NeuralNetworks::GetN_valid() const { return nnParas.N_valid; }
unsigned NeuralNetworks::GetN_test() const { return nnParas.N_test; }
unsigned NeuralNetworks::GetN_class() const { return nnParas.n_class; }



void NeuralNetworks::Initialize() {

    nnParas.n_layers = 4;
    layers = new Layer* [nnParas.n_layers];
    layers[0] = new ConvLayer;
    layers[1] = new PoolLayer;
//    layers[2] = new ConvLayer;
//    layers[3] = new PoolLayer;
    layers[2] = new FullyConnectedLayer;
    layers[3] = new SoftmaxLayer;

    arma::ivec input_dimension(3);
    input_dimension(0) = 28;
    input_dimension(1) = 28;
    input_dimension(2) = 1;

    arma::ivec filter_dimension1(3);
    filter_dimension1(0) = 5;
    filter_dimension1(1) = 5;
    filter_dimension1(2) = 6;

    arma::ivec pooling_size(2);
    pooling_size(0) = 2;
    pooling_size(1) = 2;

    arma::ivec filter_dimension2(3);
    filter_dimension2(0) = 5;
    filter_dimension2(1) = 5;
    filter_dimension2(2) = 10;

    arma::ivec hidden(3);
    hidden(0) = 100;
    hidden(1) = 1;
    hidden(2) = 1;

    arma::ivec num_class (3);
    num_class(0) = 10;
    num_class(1) = 1;
    num_class(2) = 1;

    layers[0]->Initialize_Layer("Convolutional Layer 1", input_dimension, filter_dimension1, Sigmoid, 0.);
    layers[1]->Initialize_Layer("Pooling Layer 1", layers[0]->output_dimension, pooling_size, None, 0.);
//    layers[2]->Initialize_Layer("Convolutional Layer 2", layers[1]->output_dimension, filter_dimension2, Sigmoid, 0.);
//    layers[3]->Initialize_Layer("Pooling Layer 2", layers[2]->output_dimension, pooling_size, None, 0.);
    layers[2]->Initialize_Layer("Fully Connected Layer 1", layers[1]->output_dimension, hidden, Sigmoid, 0.);
    layers[3]->Initialize_Layer("Softmax Layer", layers[2]->output_dimension, num_class, Softmax, 0.);
}



void NeuralNetworks::PrintWeights() const {
    for (unsigned l=0; l<nnParas.n_layers; l++)
        layers[l]->PrintWeight();
}

void NeuralNetworks::PrintBiases() const {
    for (unsigned l=0; l<nnParas.n_layers; l++)
        layers[l]->PrintBias();
}

void NeuralNetworks::PrintResults() const {
    for (unsigned l=0; l<nnParas.n_layers; l++) {
        layers[l]->PrintWeight();
        layers[l]->PrintBias();
    }
}


void NeuralNetworks::WriteResults(const string& filename) const {
    layers[0]->WriteResults(filename);
    for (unsigned l=1; l<nnParas.n_layers; l++) {
        layers[l]->WriteResults(filename, true);
    }
}


void NeuralNetworks::NamingFile(string& filename) {
    stringstream ss;
    for (unsigned l=0; l<nnParas.n_layers; l++) {
        if ( layers[l]->layerType == Conv ) filename += "Conv";
        if ( layers[l]->layerType == Pool ) filename += "Pool";
        if ( layers[l]->layerType == FC ) filename += "FC";
        if ( layers[l]->layerType == Soft ) filename += "Soft";
        ss << l + 1;
        filename += ss.str();    ss.str("");
    }
    filename += "lr";
    ss << nnParas.learningRate;
    filename += ss.str();    ss.str("");

    filename += "rg";
    ss << nnParas.regularization;
    filename += ss.str();    ss.str("");

    filename += "mo";
    ss << nnParas.momentum;
    filename += ss.str();    ss.str("");

    filename += ".txt";
}

void NeuralNetworks::NamingFileStep(string& filename, const unsigned& step) {
    stringstream ss;
    for (unsigned l=0; l<nnParas.n_layers; l++) {
        if ( layers[l]->layerType == Conv ) filename += "Conv";
        if ( layers[l]->layerType == Pool ) filename += "Pool";
        if ( layers[l]->layerType == FC ) filename += "FC";
        if ( layers[l]->layerType == Soft ) filename += "Soft";
        ss << l + 1;
        filename += ss.str();    ss.str("");
    }
    filename += "lr";
    ss << nnParas.learningRate;
    filename += ss.str();    ss.str("");

    filename += "rg";
    ss << nnParas.regularization;
    filename += ss.str();    ss.str("");

    filename += "mo";
    ss << nnParas.momentum;
    filename += ss.str();    ss.str("");

    filename += "step";
    ss << step;
    filename += ss.str();    ss.str("");

    filename += ".txt";
}





template<typename dataType>
void NeuralNetworks::Training(df::DataFrame<dataType>& data, const unsigned& step) {

    string parafile = "conv.parameter.";
    NamingFile(parafile);
    WriteParameters(parafile);

    for (unsigned epoch=0; epoch<nnParas.maxEpoch; epoch++) {
        cout << "epochs: " << epoch << endl;
        double remain_epoch = (double) epoch / (double) nnParas.maxEpoch * 100;
        cout << "remaining epochs ratio: " << remain_epoch << "%" << endl << endl;
        TrainingOneStep(data);
    }
    
    if ( step % 10 == 0 ) {
        string resfile = "conv.result.";
        NamingFileStep(resfile, step);
        WriteResults(resfile);
    }
}

template<typename dataType>
void NeuralNetworks::Training(df::DataFrame<dataType>& data, df::DataFrame<dataType>& valid, unsigned& step) {

    string parafile = "conv.parameter.";
    NamingFile(parafile);
    WriteParameters(parafile);

    for (unsigned epoch=0; epoch<nnParas.maxEpoch; epoch++) {
        cout << "epochs: " << epoch << endl;
        double remain_epoch = (double) epoch / (double) nnParas.maxEpoch * 100;
        cout << "remaining epochs ratio: " << remain_epoch << "%" << endl << endl;
        TrainingOneStep(data, valid);
    }
    
    if ( step % 10 == 0 ) {
        string resfile = "conv.result.";
        NamingFileStep(resfile, step);
        WriteResults(resfile);
    }
}



template<typename dataType>
void NeuralNetworks::TrainingOneStep(df::DataFrame<dataType>& data) {

    arma::field<Vector> minibatch;
    MiniBathces(minibatch);

    double error = 0.0;
    for (unsigned n=0; n<minibatch.size(); n++)
        TrainingMiniBatch(data, minibatch(n), error);
    error /= (double) nnParas.N_train;

    string errorfile = "conv.error.";
    NamingFile(errorfile);
    WriteError(errorfile, error);
}


template<typename dataType>
void NeuralNetworks::TrainingOneStep(df::DataFrame<dataType>& data, df::DataFrame<dataType>& valid) {

    arma::field<Vector> minibatch;
    MiniBathces(minibatch);

    double error = 0.0;
    for (unsigned n=0; n<minibatch.size(); n++)
        TrainingMiniBatch(data, minibatch(n), error);
    error /= (double) nnParas.N_train;

    //  validation test
    double valid_error = 0.0;
    double valid_accuracy = 0.0;
    Validation(valid, valid_error, valid_accuracy);


    string errorfile = "conv.error.";
    NamingFile(errorfile);
    WriteError(errorfile, error, valid_error, valid_accuracy);
}



void NeuralNetworks::MiniBathces(arma::field<Vector>& minibatch) {

    boost::random::uniform_real_distribution<> uniform_real_dist(0, 1);                 //  Choose a distribution
    boost::random::variate_generator<boost::mt19937 &,
        boost::random::uniform_real_distribution<> > urnd(rng, uniform_real_dist);      //  link the Generator to the distribution

    Vector rand_data(nnParas.N_train);
    for (unsigned n=0; n<nnParas.N_train; n++)
        rand_data(n) = urnd();
    arma::uvec shuffleindex = sort_index(rand_data);

    unsigned n_minibatch = (unsigned) (nnParas.N_train / nnParas.minibatchSize);
    unsigned remainder = nnParas.N_train % nnParas.minibatchSize;
    if ( remainder != 0 ) {
        n_minibatch++;
        minibatch.set_size(n_minibatch);
        unsigned index = 0;
        for (unsigned n=0; n<n_minibatch-1; n++) {
            minibatch(n).set_size(nnParas.minibatchSize);
            for (unsigned j=0; j<nnParas.minibatchSize; j++)
                minibatch(n)(j) = shuffleindex(index++);
        }
        minibatch(n_minibatch-1).set_size(remainder);
        for (unsigned j=0; j<remainder; j++)
            minibatch(n_minibatch-1)(j) = shuffleindex(index++);
    }
    else {
        minibatch.set_size(n_minibatch);
        unsigned index = 0;
        for (unsigned n=0; n<n_minibatch; n++) {
            minibatch(n).set_size(nnParas.minibatchSize);
            for (unsigned j=0; j<nnParas.minibatchSize; j++)
                minibatch(n)(j) = shuffleindex(index++);
        }
    }
}


void NeuralNetworks::InitializeDeltaParameters() {

    for (unsigned l=0; l<nnParas.n_layers; l++)
        layers[l]->InitializeDeltaParameters();
}


template<typename dataType>
void NeuralNetworks::TrainingMiniBatch(df::DataFrame<dataType>& data, const Vector& minibatch, double& error) {

    InitializeDeltaParameters();

    for (unsigned n=0; n<minibatch.size(); n++) {

        //  Pick one record
        arma::Row<dataType> x = data.GetDataRow(minibatch(n));
        arma::irowvec t = data.GetTargetMatrixRow(minibatch(n));

        //  FeedForward learning
        FeedForward(x);

        //  Error estimation
        CostFunction(error, t.t());

        //  Error Back Propagation
        BackPropagation(x, t.t());
    }

    //  Update Parameters
    UpdateParameter(minibatch.size());
}




Cube NeuralNetworks::ReshapeVectortoCube(const Vector& input_vec, const arma::ivec& input_dim) {
    
    Cube input(input_dim(0), input_dim(1), input_dim(2));
    for (unsigned k=0; k<input_dim(2); k++)
        for (unsigned j=0; j<input_dim(1); j++)
            for (unsigned i=0; i<input_dim(0); i++)
                input(i, j, k) = input_vec(i + j*input_dim(0) + k*input_dim(0)*input_dim(1));
//                input(i, j, k) = input_vec(i*input_dim(1) + j + k*input_dim(0)*input_dim(1));

    return input;
}

Cube NeuralNetworks::ReshapeVectortoCube(const Vector& input_vec, const unsigned& height, const unsigned& width, const unsigned& depth) {
    
    Cube input(height, width, depth);
    for (unsigned k=0; k<depth; k++)
        for (unsigned j=0; j<width; j++)
            for (unsigned i=0; i<height; i++)
                input(i, j, k) = input_vec(i + j*height+ k*height*width);
//                input(i, j, k) = input_vec(i*width + j + k*height*width);

    return input;
}

Vector NeuralNetworks::ReshapeCubetoVector(const Cube& input_cube) {
    
    unsigned height = input_cube.n_rows;
    unsigned width = input_cube.n_cols;
    unsigned depth = input_cube.n_slices;
    Vector input(height * width * depth);

    for (unsigned k=0; k<depth; k++)
        for (unsigned j=0; j<width; j++)
            for (unsigned i=0; i<height; i++)
                input(i + j*height + k*height*width) = input_cube(i, j, k);
//                input(i*width + j + k*height*width) = input_cube(i, j, k);

    return input;
}






template<typename dataType>
void NeuralNetworks::FeedForward(const arma::Row<dataType>& x) {

    layers[0]->Forward(ReshapeVectortoCube(x.t(), layers[0]->input_dimension));
    for (unsigned l=1; l<nnParas.n_layers; l++)
        layers[l]->Forward(layers[l-1]->activation);
}


void NeuralNetworks::CostFunction(double& error, const arma::ivec& t) {

    Vector act = ReshapeCubetoVector(layers[nnParas.n_layers-1]->activation);
    if ( nnParas.cost != CrossEntropy )
        error += arma::dot(t - act, t - act) * 0.5;
    else
        error += - arma::dot(t, log(act)) - arma::dot(1. - t, log(1. - act));
}

template<typename dataType>
void NeuralNetworks::Validation(df::DataFrame<dataType>& valid, double& error, double& accuracy) {

    for (unsigned n=0; n<nnParas.N_valid; n++) {

        //  Pick one record
        arma::Row<dataType> x = valid.GetDataRow(n);
        arma::irowvec t = valid.GetTargetMatrixRow(n);

        //  FeedForward learning
        FeedForward(x);

        //  Error estimation
        CostFunction(error, t.t());

        //  accuracy estimation
        arma::uword index;
        Vector act = ReshapeCubetoVector(layers[nnParas.n_layers-1]->activation);
        double max_value = act.max(index);
        if ( index != valid.GetTarget(n) ) accuracy += 1.;
    }
    error /= (double) nnParas.N_valid;
    accuracy /= (double) nnParas.N_valid;
}


template<typename dataType>
void NeuralNetworks::BackPropagation(const arma::Row<dataType>& x, const arma::ivec& t) {

    layers[nnParas.n_layers-1]->Backward(t, nnParas.cost);
    for (unsigned l=nnParas.n_layers-2; l>0; l--)
        layers[l]->Backward(*(layers[l+1]));
    layers[0]->Backward(*(layers[1]));

//  Cumulation of delta_weight and delta_bias in minibatch
    layers[0]->Cumulation(ReshapeVectortoCube(x.t(), layers[0]->input_dimension));
    for (unsigned l=1; l<nnParas.n_layers; l++)
        layers[l]->Cumulation(layers[l-1]->activation);
}

void NeuralNetworks::UpdateParameter(const unsigned& minibatchSize) {

    for (unsigned l=0; l<nnParas.n_layers; l++)
        //  Without Momentum
        //layers[l]->WithoutMomentum(nnParas.learningRate, nnParas.regularization, nnParas.N_train, nnParas.minibatchSize);
        //  Momentum
        //layers[l]->Momentum(nnParas.learningRate, nnParas.regularization, nnParas.momentum, nnParas.N_train, nnParas.minibatchSize);
        //  Nesterov Momentum
        layers[l]->NesterovMomentum(nnParas.learningRate, nnParas.regularization, nnParas.momentum,
                nnParas.N_train, nnParas.minibatchSize);
}

void NeuralNetworks::WriteError(const string& filename, const double& error) {
    ofstream fsave(filename.c_str(), fstream::out | fstream::app);
    fsave << error << endl;
    fsave.close();
}

void NeuralNetworks::WriteError(const string& filename, const double& error, const double& valid_error, const double& valid_accuracy) {
    ofstream fsave(filename.c_str(), fstream::out | fstream::app);
    fsave << error << " " << valid_error << " " << valid_accuracy << endl;
    fsave.close();
}



template<typename dataType>
void NeuralNetworks::Test(df::DataFrame<dataType>& data, arma::uvec& predict, unsigned& step) {

    for (unsigned n=0; n<data.GetN(); n++) {
        //  Pick one record
        arma::Row<dataType> x = data.GetDataRow(n);

        //  FeedForward learning
        FeedForward(x);

        //  Predict class
        arma::uword index;
        Vector act = ReshapeCubetoVector(layers[nnParas.n_layers-1]->activation);
        double max_value = act.max(index);
        predict(n) = index;
    }

    string predfile = "conv.predict.";
    NamingFileStep(predfile, step);

    ofstream fsave(predfile.c_str());
    predict.raw_print(fsave, "predeict test data");
    fsave.close();
}





}





#endif
