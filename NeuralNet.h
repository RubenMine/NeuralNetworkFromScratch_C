#ifndef NEURALNET_H
#define NEURALNET_H
#include "mat.h"

typedef struct NeuronLayer{
    Matrix* weights; //row are the neurons, cols are the input 
    Matrix* bias;
    Matrix* out;

    Matrix* w_grad;
    Matrix* b_grad;
}NeuronLayer;

typedef struct NeuralNetwork{
    NeuronLayer* hiddenLayers;
    int n_of_layer;
    int n_input;

    Matrix* temp; // matrice di appoggio che uso per fare calcoli 
}NeuralNetwork;


NeuralNetwork* nn_create(int n_input, int n_layers, int* layer_sizes);
void nn_init_Layer(NeuronLayer* nl, int n_input, int n_neurons);
void nn_free(NeuralNetwork* nn);
void nn_print(NeuralNetwork* nn);

Matrix* nn_evaluate(NeuralNetwork* nn, Matrix* x);
float nn_cost(NeuralNetwork* nn, Matrix* data, Matrix* y);
void nn_finite_diff(NeuralNetwork* nn, float eps, Matrix* data, Matrix* y);
void nn_apply_gradient(NeuralNetwork* nn, float rate);
void nn_train(NeuralNetwork* nn, Matrix* data, Matrix* y, int epoch, int eps, int rate);

#endif // NEURALNET_H