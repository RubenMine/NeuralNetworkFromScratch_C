#include <stdio.h>
#include <stdlib.h>


// Implementazione Matrice con memoria adiacente
typedef struct Matrix{
    int rows;
    int cols;
    float* data;
}Matrix;

Matrix* mat_create(int n, int m) {
    Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
    mat->rows = n;
    mat->cols = m;
    mat->data = (float*)malloc(n * m * sizeof(float));
    return mat;
}

Matrix* mat_dot(Matrix* M1, Matrix* M2) {
    if (M1->cols != M2->rows) {
        printf("Error: Incompatible matrix dimensions\n");
        return NULL;
    }

    Matrix* result = mat_create(M1->rows, M2->cols);

    for (int i = 0; i < M1->rows; i++) {
        for (int j = 0; j < M2->cols; j++) {
            float sum = 0;
            for (int k = 0; k < M1->cols; k++) {
                sum += M1->data[i * M1->cols + k] * M2->data[k * M2->cols + j];
            }
            result->data[i * result->cols + j] = sum;
        }
    }

    return result;
}

Matrix* mat_sum(Matrix* M1, Matrix* M2) {
    if (M1->rows != M2->rows || M1->cols != M2->cols) {
        printf("Error: Incompatible matrix dimensions\n");
        return NULL;
    }

    Matrix* result = mat_create(M1->rows, M1->cols);

    for (int i = 0; i < M1->rows; i++) {
        for (int j = 0; j < M1->cols; j++) {
            result->data[i * result->cols + j] = M1->data[i * M1->cols + j] + M2->data[i * M2->cols + j];
        }
    }

    return result;
}

Matrix* mat_sub(Matrix* M1, Matrix* M2) {
    if (M1->rows != M2->rows || M1->cols != M2->cols) {
        printf("Error: Incompatible matrix dimensions\n");
        return NULL;
    }

    Matrix* result = mat_create(M1->rows, M1->cols);

    for (int i = 0; i < M1->rows; i++) {
        for (int j = 0; j < M1->cols; j++) {
            result->data[i * result->cols + j] = M1->data[i * M1->cols + j] - M2->data[i * M2->cols + j];
        }
    }

    return result;
}

float mat_get(Matrix* M, int row, int col) {
    if (row >= M->rows || col >= M->cols) {
        printf("Error: Invalid row or column index\n");
        return 0.0;
    }

    return M->data[row * M->cols + col];
}

void mat_set(Matrix* M, int row, int col, float value) {
    if (row >= M->rows || col >= M->cols) {
        printf("Error: Invalid row or column index\n");
        return;
    }

    M->data[row * M->cols + col] = value;
}


Matrix* mat_extract_row(Matrix* M, int row) {
    if (row >= M->rows) {
        printf("Error: Invalid row index\n");
        return NULL;
    }

    Matrix* result = mat_create(1, M->cols);

    for (int j = 0; j < M->cols; j++) {
        result->data[j] = M->data[row * M->cols + j];
    }

    return result;
}

Matrix* mat_transpose(Matrix* M) {
    Matrix* result = mat_create(M->cols, M->rows);

    for (int i = 0; i < M->rows; i++) {
        for (int j = 0; j < M->cols; j++) {
            result->data[j * result->cols + i] = M->data[i * M->cols + j];
        }
    }

    return result;
}

void mat_print(Matrix* mat) {
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            printf("%.2f ", mat->data[i * mat->cols + j]);
        }
        printf("\n");
    }
}

void mat_free(Matrix* mat) {
    free(mat->data);
    free(mat);
}


Matrix* mat_extract_sample(Matrix* M, int start_row, int start_col, int step){
    if (M->rows <= start_row || M->cols <= start_col) {
        printf("Error: Incompatible matrix dimensions\n");
        return NULL;
    }

    //ALLOCO PIU MEMORIA DI QUELLA CHE SERVE
    Matrix* result = mat_create(M->rows, 1);
    int j = 0;
    for(int i = start_col+start_row; i < (M->cols+M->rows); i+=step){
        result->data[j] = M->data[i];
        j++;
    }

    return result;
}

Matrix* mat_scalar_mult(Matrix* M, float scalar) {
    Matrix* result = mat_create(M->rows, M->cols);

    for (int i = 0; i < M->rows; i++) {
        for (int j = 0; j < M->cols; j++) {
            result->data[i * result->cols + j] = M->data[i * M->cols + j] * scalar;
        }
    }

    return result;
}



Matrix* mat_from_array(float* arr, int rows, int cols) {
    Matrix* mat = mat_create(rows, cols);

    for (int i = 0; i < rows; i++) {
        mat->data[i * mat->cols] = arr[i];
    }

    return mat;
}

Matrix* mat_from_arrMat(float matrix[16][4], int rows, int cols) {
    Matrix* mat = mat_create(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float v = matrix[i][j];
            mat->data[i * (mat->cols) + j] = v;
        }
    }

    return mat;
}