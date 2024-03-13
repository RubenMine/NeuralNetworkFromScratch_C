#include "NeuralNet.c"
#include <math.h>

float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

void mat_sigmoid(Matrix* x){
    for(int i = 0; i<x->rows; i++){
        for(int j = 0; j<x->cols; j++){
            float v = mat_get(x, i, j);
            mat_set(x, i, j, sigmoid(v));
        }
    }
}

float cost(NeuralNetwork* nn, Matrix* data, Matrix* y){
    // each column of Matrix x is a sample
    // rows is number of value in one sample
    float totalErr = 0;
    for(int i = 0; i<data->rows; i++){
        //printf("Sample: %d\n", i);
        //mat_print(mat_transpose(mat_extract_row(y, i)));
        Matrix* sample_x = mat_transpose(mat_extract_row(data, i));
        Matrix* sample_y = mat_transpose(mat_extract_row(y, i));

        Matrix* err = mat_sub(evaluate(nn, sample_x), sample_y);
        err = mat_dot(err, mat_transpose(err));
        totalErr += *err->data;
        //x = mat_sum(mat_dot(l.weights, x), l.bias);
    }
    return totalErr/data->cols;
}

void finite_diff(NeuralNetwork* nn, float eps, Matrix* data, Matrix* y){
    float initial_err = cost(nn, data, y);
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
                mat_set(nl->w_grad, j, z, ((cost(nn, data, y) - initial_err)/eps));
                //printf("derr: %f\n",(cost(nn, data, y) - initial_err));
                mat_set(nl->weights, j, z, saved);
            }
        }
        //mat_print(nl->w_grad);
        //printf("\n");
    }
}

void apply_gradient(NeuralNetwork* nn, float rate){
    for(int i = 0; i<nn->n_of_layer; i++){
        NeuronLayer* nl = &(nn->hiddenLayers[i]);
        //nl->weights = mat_sub(nl->w_grad, mat_scalar_mult(nl->weights, rate));
        nl->weights = mat_sub(nl->w_grad, mat_scalar_mult(nl->weights, rate));
    }
}

void train(NeuralNetwork* nn, Matrix* data, Matrix* y, int epoch, int eps, int rate){
    printf("Pre-train MSE: %f\n", cost(nn, data, y));
    
    for(int i = 0; i<epoch; i++){
        finite_diff(nn, 1e-3, data, y);
        apply_gradient(nn, 1);
    }
    printf("Post-train MSE: %f\n", cost(nn, data, y));
}





/*
void backprop(NeuralNetwork* nn){
    for(int i = nn->n_of_layer - 1; i >= 0; i--){
        NeuronLayer* current_layer = &(nn->layers[i]);
        NeuronLayer* next_layer = (i < nn->n_of_layer - 1) ? &(nn->layers[i+1]) : NULL;

        // Calculate the error gradient for the current layer
        if(next_layer != NULL){
            Matrix* next_delta = mat_dot(mat_transpose(next_layer->weights), current_layer->delta);
            Matrix* current_activation_derivative = activation_derivative(current_layer->activation, current_layer->output);
            current_layer->delta = mat_mul(next_delta, current_activation_derivative);
            mat_free(next_delta);
            mat_free(current_activation_derivative);
        }

        // Update the weights and biases of the current layer
        Matrix* current_activation = mat_transpose(current_layer->output);
        Matrix* delta_weights = mat_dot(current_layer->delta, current_activation);
        Matrix* delta_biases = mat_copy(current_layer->delta);

        mat_scalar_mul(delta_weights, -nn->learning_rate);
        mat_scalar_mul(delta_biases, -nn->learning_rate);

        mat_add(current_layer->weights, delta_weights);
        mat_add(current_layer->biases, delta_biases);

        mat_free(current_activation);
        mat_free(delta_weights);
        mat_free(delta_biases);
    }
}
*/


/*
#define train_count sizeof(train_x)/sizeof(train_x[0])
float error(NeuralNetwork* nn, int print){
    float err = 0.0f;
    for(int i = 0; i < train_count; i++){
        float* x = (float*)train_x[i];
        float* p_y = evaluate(nn, x);
        float single_err = p_y[0] - train_y[i];
        err += single_err*single_err;
        //printf("Input: ");
        //printArray(x, 2);
        if(print){printf("Expected: %f, Predicted: %f, Error: %f\n", train_y[i], *p_y, single_err);}
    }
    return (err / train_count);
}

NeuronLayer* copyNeuronLayer(NeuronLayer* layer) {
    NeuronLayer* copy = (NeuronLayer*)malloc(sizeof(NeuronLayer));
    copy->n_of_neurons = layer->n_of_neurons;
    copy->n_of_input = layer->n_of_input;
    copy->w_neurons = (float**)malloc(copy->n_of_neurons * sizeof(float*));
    copy->b_neurons = (float*)malloc(copy->n_of_neurons * sizeof(float));

    for (int i = 0; i < copy->n_of_neurons; i++) {
        copy->w_neurons[i] = (float*)malloc(copy->n_of_input * sizeof(float));
        for (int j = 0; j < copy->n_of_input; j++) {
            copy->w_neurons[i][j] = layer->w_neurons[i][j];
        }
        copy->b_neurons[i] = layer->b_neurons[i];
    }

    return copy;
}


NeuronLayer* backprop(NeuralNetwork* nn, float initial_err, float eps, int i){
    NeuronLayer* nl = &(nn->layers[i]);
    float saved;
    NeuronLayer* copy = copyNeuronLayer(nl);

    for(int j = 0; j < nl->n_of_neurons; j++){
        float* ow = nl->w_neurons[j];
        for(int z = 0; z < nl->n_of_input; z++){
            saved = nl->w_neurons[j][z];
            nl->w_neurons[j][z] += eps;
            float dder = (error(nn, 0) - initial_err)/eps;
            nl->w_neurons[j][z] = saved;
            copy->w_neurons[j][z] -= dder;
        }
    }

    if(i < nn->n_of_layer-1){ backprop(nn, initial_err, eps, i-1); }

    nn->layers[i] = *copy;
}


float train(NeuralNetwork* nn, int epoch, float rate){
    float eps = 1e-3;
    for(int i = 0; i<epoch; i++){
        if(i==0 || i == epoch-1){printf("Error:%f\n", error(nn, 1));}
        float err = error(nn, 0);
        backprop(nn, err, eps, nn->n_of_layer-1);
    }
    return error(nn, 0);
}
*/

int main(){
    float trainx[][4] = {
        {0, 0, 0, 0},
        {1, 0, 0, 0},
        {0, 1, 0, 0},
        {1, 1, 0, 0},
        {0, 0, 1, 0},
        {1, 0, 1, 0},
        {0, 1, 1, 0},
        {1, 1, 1, 0},
        {0, 0, 0, 1},
        {1, 0, 0, 1},
        {0, 1, 0, 1},
        {1, 1, 0, 1},
        {0, 0, 1, 1},
        {1, 0, 1, 1},
        {0, 1, 1, 1},
        {1, 1, 1, 1}
    };
    float trainy[] = {0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};


    srand(time(0));
    int n_of_node_ly[3] = {4, 2, 1};
    NeuralNetwork* nn = create_NeuralNet(4, 3, n_of_node_ly);

    Matrix* train_x = mat_from_arrMat(trainx, 16, 4);
    Matrix* train_y = mat_from_array((float*)trainy, 16, 1);
    //fillWithRandomValue(train_x, 5.0f);
    //fillWithRandomValue(train_y, 20.0f);

    printf("\n\nInput: \n");
    mat_print(train_x);
    fflush(stdout);
    printf("Predicted: \n");
    for(int i = 0; i<4; i++)
        mat_print(evaluate(nn, mat_transpose(mat_extract_row(train_x, i))));
    //printf("Error:%f\n", cost(nn, train_x, train_y));
    
    train(nn, train_x, train_y, 10000, 1e-3, 1);
    printf("\nPredicted: \n");
    for(int i = 0; i<16; i++)
        mat_print(evaluate(nn, mat_transpose(mat_extract_row(train_x, i))));
    print_NeuralNet(nn);
    //print_NeuralNet(nn);
    /*
    //float x[2] = {1.5f, 3.0f};
    //float* res = evaluate(nn, x);
    printf("Error: %f\n", error(nn, 0));
    //train(nn, 1000, 1e-1);
    printf("Error: %f\n", train(nn, 1000, 1e-1));
    printf("x: ");
    */
    return 0;
}
