#include <stdio.h>
#include <stdlib.h>
#include "time.h"
#include <math.h>
#include "util.c"

typedef struct NeuronLayer{
    Matrix* weights; //row are the neurons, cols are the input 
    Matrix* bias;
    Matrix* w_grad;
    Matrix* b_grad;
}NeuronLayer;

typedef struct NeuralNetwork{
    NeuronLayer* hiddenLayers;
    int n_of_layer;
    int n_input;
}NeuralNetwork;


void init_Layer(NeuronLayer* nl, int n_input, int n_neurons){
    nl->weights = mat_create(n_neurons, n_input);
    nl->w_grad = mat_create(n_neurons, n_input);
    nl->bias = mat_create(n_neurons, 1);
    nl->b_grad = mat_create(n_neurons, 1);
    fillWithRandomValue(nl->bias, 0.0f);
    fillWithRandomValue(nl->b_grad, 0.0f);
    fillWithRandomValue(nl->weights, 3.0f);
    fillWithRandomValue(nl->w_grad, 0.0f);
}

NeuralNetwork* create_NeuralNet(int n_input, int n_layers, int* layer_sizes){
    NeuralNetwork* nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    nn->n_of_layer = n_layers;
    nn->n_input = n_input; 

    nn->hiddenLayers = (NeuronLayer*)malloc(n_layers * sizeof(NeuronLayer));
    
    for(int i = 0; i<n_layers; i++){
        NeuronLayer* nl = &(nn->hiddenLayers[i]);
        
        if(i==0){init_Layer(nl, n_input, layer_sizes[0]);}
        else{init_Layer(nl, layer_sizes[i-1], layer_sizes[i]);}
    }
    print_NeuralNet(nn);
    return nn;
}
    
void print_NeuralNet(NeuralNetwork* nn) {
    for (int i = 0; i < nn->n_of_layer; i++) {
        NeuronLayer layer = nn->hiddenLayers[i];
        printf("Layer %d:\n", i + 1);
        printf("Weights:\n");
        mat_print(layer.weights);
        
        printf("Biases:\n");
        mat_print(layer.bias);
        printf("\n");
    }
}

Matrix* evaluate(NeuralNetwork* nn, Matrix* x){
    for(int i = 0; i<nn->n_of_layer; i++){
        NeuronLayer l = nn->hiddenLayers[i];
        x = mat_sum(mat_dot(l.weights, x), l.bias);
        mat_sigmoid(x);
    }
    return x;
}