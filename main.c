#include "NeuralNet.c"

int main(){
    float train_arr[][5] = {
        {0, 0, 0, 0, 0},
        {1, 0, 0, 0, 0},
        {0, 1, 0, 0, 0},
        {1, 1, 0, 0, 1},
        {0, 0, 1, 0, 1},
        {1, 0, 1, 0, 1},
        {0, 1, 1, 0, 1},
        {1, 1, 1, 0, 1},
        {0, 0, 0, 1, 0},
        {1, 0, 0, 1, 0},
        {0, 1, 0, 1, 0},
        {1, 1, 0, 1, 0},
        {0, 0, 1, 1, 0},
        {1, 0, 1, 1, 0},
        {0, 1, 1, 1, 0},
        {1, 1, 1, 1, 0}
    };

    srand(time(0));
    int arch[3] = {2, 2, 1};
    NeuralNetwork* nn = nn_create(4, 3, arch);

    Matrix* dataset = mat_create(16, 5);
    mat_copy_data_from_arr(dataset, (float**)train_arr);
    Matrix* tx = mat_extract_submat(dataset, 16, 4, 0, 0);
    Matrix* ty = mat_extract_col(dataset, 4);


    printf("\n\nInput: \n");
    mat_print(tx);
    
    nn_train(nn, tx, ty, 10000, 1e-3, 1);

    printf("\nPredicted: \n");
    for(int i = 0; i<16; i++)
        mat_print(nn_evaluate(nn, mat_transpose(mat_extract_row(tx, i))));


    nn_free(nn);

    return 0;
}