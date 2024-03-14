#include "mat.c"
#include "NeuralNet.h"

NeuralNetwork* nn_create(int n_input, int n_layers, int* layer_sizes){
    NeuralNetwork* nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    nn->n_of_layer = n_layers;
    nn->n_input = n_input; 

    nn->hiddenLayers = (NeuronLayer*)malloc(n_layers * sizeof(NeuronLayer));
    
    for(int i = 0; i<n_layers; i++){
        NeuronLayer* nl = &(nn->hiddenLayers[i]);
        
        if(i==0){nn_init_layer(nl, n_input, layer_sizes[0]);}
        else{nn_init_layer(nl, layer_sizes[i-1], layer_sizes[i]);}
    }
    nn_print(nn);
    return nn;
}

void nn_init_layer(NeuronLayer* nl, int n_input, int n_neurons){
    nl->weights = mat_create(n_neurons, n_input);
    nl->bias = mat_create(n_neurons, 1);
    nl->out = mat_create(n_neurons, 1);

    nl->w_grad = mat_create(n_neurons, n_input);
    nl->b_grad = mat_create(n_neurons, 1);
    mat_random_fill(nl->bias, 0.0f);
    mat_random_fill(nl->b_grad, 0.0f);
    mat_random_fill(nl->weights, 3.0f);
    mat_random_fill(nl->w_grad, 0.0f);
}

void nn_free(NeuralNetwork* nn) {
    for (int i = 0; i < nn->n_of_layer; i++) {
        NeuronLayer* layer = &(nn->hiddenLayers[i]);
        mat_free(layer->weights);
        mat_free(layer->bias);
        mat_free(layer->out);
        mat_free(layer->w_grad);
        mat_free(layer->b_grad);
    }
    free(nn->hiddenLayers);
    free(nn);
}

void nn_print(NeuralNetwork* nn) {
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

//ritorna una matrice di dimensioni dell'output finale
Matrix* nn_evaluate(NeuralNetwork* nn, Matrix* x){
    for(int i = 0; i<nn->n_of_layer; i++){
        NeuronLayer l = nn->hiddenLayers[i];
        mat_dot(l.weights, x, l.out);
        mat_sum(l.out, l.bias);
        mat_apply_function(l.out, &sigmoid);
        x = l.out;
    }
    return x;
}

float nn_cost(NeuralNetwork* nn, Matrix* data, Matrix* y){
    // each column of Matrix x is a sample
    // rows is number of value in one sample
    Matrix* total_err = mat_create(1, 1);
    float totalErr = 0;
    for(int i = 0; i<data->rows; i++){
        //printf("Sample: %d\n", i);
        //mat_print(mat_transpose(mat_extract_row(y, i)));
        Matrix* sample_x = mat_transpose(mat_extract_row(data, i));
        Matrix* sample_y = mat_transpose(mat_extract_row(y, i));

        nn->temp = nn_evaluate(nn, sample_x);
        mat_sub(nn->temp, sample_y);
        mat_dot(nn->temp, mat_transpose(nn->temp), total_err);
        totalErr += *total_err->data;
        //x = mat_sum(mat_dot(l.weights, x), l.bias);
    }
    return totalErr/data->cols;
}

void nn_finite_diff(NeuralNetwork* nn, float eps, Matrix* data, Matrix* y){
    float initial_err = nn_cost(nn, data, y);
    float saved = 0;

    // for each layer
    for(int i = 0; i < nn->n_of_layer; i++){
        NeuronLayer* nl = &(nn->hiddenLayers[i]);

        // for each node in layer
        for(int j = 0; j < nl->weights->rows; j++){

            // for each weight in node
            for(int z = 0; z < nl->weights->cols; z++){
                saved = mat_get(nl->weights, j, z); //weight value
                mat_set(nl->weights, j, z, mat_get(nl->weights, j, z) + eps);
                mat_set(nl->w_grad, j, z, ((nn_cost(nn, data, y) - initial_err)/eps));
                //printf("derr: %f\n",(cost(nn, data, y) - initial_err));
                mat_set(nl->weights, j, z, saved);
            }
        }
    }
}

void nn_apply_gradient(NeuralNetwork* nn, float rate){
    for(int i = 0; i<nn->n_of_layer; i++){
        NeuronLayer* nl = &(nn->hiddenLayers[i]);
        //nl->weights = mat_sub(nl->w_grad, mat_scalar_mult(nl->weights, rate));
        //w -= w - r*g
        mat_scalar_mult(nl->w_grad, rate);
        mat_sub(nl->weights, nl->w_grad);
        //nl->weights = mat_sub(nl->w_grad, mat_scalar_mult(nl->weights, rate));
    }
}

void nn_train(NeuralNetwork* nn, Matrix* data, Matrix* y, int epoch, int eps, int rate){
    printf("Pre-train MSE: %f\n", nn_cost(nn, data, y));
    
    for(int i = 0; i<epoch; i++){
        nn_finite_diff(nn, 1e-3, data, y);
        nn_apply_gradient(nn, 1);
    }
    printf("Post-train MSE: %f\n", nn_cost(nn, data, y));
}
