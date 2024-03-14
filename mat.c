#include "mat.h"
#include "util.c"
#include <math.h>
#include <assert.h>
#include <stdbool.h>

Matrix* mat_create(int n, int m) {
    Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
    mat->data = (float*)malloc(n * m * sizeof(float));
    mat->rows = n;
    mat->cols = m;
    mat->offset = m;
    return mat;
}

void mat_free(Matrix* mat) {
    free(mat->data);
    free(mat);
}

float mat_get(Matrix* M, int row, int col) {
    bool mat_check = !(row >= M->rows || col >= M->cols);
    assert(mat_check);

    return M->data[(row * M->offset) + col];
}

void mat_set(Matrix* M, int row, int col, float value) {
    bool mat_check = !(row >= M->rows || col >= M->cols);
    assert(mat_check);

    M->data[(row * M->offset) + col] = value;
}

void mat_print(Matrix* M) {
    for (int i = 0; i < M->rows; i++) {
        for (int j = 0; j < M->cols; j++) {
            printf("%.2f ", mat_get(M, i, j));
        }
        printf("\n");
    }
}

void mat_sum(Matrix* M1, Matrix* M2) {
    bool mat_check = !(M1->rows != M2->rows || M1->cols != M2->cols);
    assert(mat_check);

    for (int i = 0; i < M1->rows; i++) {
        float v = 0;
        for (int j = 0; j < M1->cols; j++) {
            v = mat_get(M1, i, j) + mat_get(M2, i, j);
            mat_set(M1, i, j, v);
            //M1->data[(i * M1->cols * M1->offset) + j] += mat_get(M2, i, j);
       }
    }
}

void mat_sub(Matrix* M1, Matrix* M2) {
    bool mat_check = !(M1->rows != M2->rows || M1->cols != M2->cols);
    assert(mat_check);

    for (int i = 0; i < M1->rows; i++) {
        float v = 0;
        for (int j = 0; j < M1->cols; j++) {
            v = mat_get(M1, i, j) - mat_get(M2, i, j);
            mat_set(M1, i, j, v);
            //M1->data[(i * M1->cols * M1->offset) + j] -= mat_get(M2, i, j);
        }
    }
}

void mat_scalar_mult(Matrix* M, float scalar) {
    for (int i = 0; i < M->rows; i++) {
        float v = 0;
        for (int j = 0; j < M->cols; j++) {
            v = mat_get(M, i, j) * scalar;
            mat_set(M, i, j, v);
        }
    }
}

void mat_dot(Matrix* M1, Matrix* M2, Matrix* res) {
    bool mat_check = !(M1->cols != M2->rows);
    assert(mat_check);

    for (int i = 0; i < M1->rows; i++) {
        for (int j = 0; j < M2->cols; j++) {
            float sum = 0;
            for (int k = 0; k < M1->cols; k++) {
                if(M1 == M2){
                    sum += mat_get(M1, i, k) * mat_get(M2, i, k);
                }else{
                    sum += mat_get(M1, i, k) * mat_get(M2, k, j);   
                }
                //sum += M1->data[i * M1->cols + k] * M2->data[k * M2->cols + j];
            }
            mat_set(res, i, j, sum); 
        }
    }
}

Matrix* mat_extract_submat(Matrix* M, int rows, int cols, int start_row, int start_col){
    // Controllo che la riga/colonna iniziale sia all'interno della mat
    // Controllo anche che la riga/colonna + rows/cols sia all'interno della mat
    bool mat_check = !(start_row >= M->rows || start_col >= M->cols || start_row + rows > M->rows || start_col + cols > M->cols);
    assert(mat_check);

    Matrix* res = (Matrix*)malloc(sizeof(Matrix));
    *res = (Matrix){
        .rows = rows,
        .cols = cols,
        .data = M->data + ((start_row * M->offset) + start_col),
        .offset = M->offset
    };

    return res;
}

Matrix* mat_extract_row(Matrix* M, int row) {
    return mat_extract_submat(M, 1, M->cols, row, 0);
}

Matrix* mat_extract_col(Matrix* M, int col) {
    return mat_extract_submat(M, M->rows, 1, 0, col);
}


// da ottimizzare
Matrix* mat_transpose(Matrix* M) {
    Matrix* result = mat_create(M->cols, M->rows);

    for (int i = 0; i < M->rows; i++) {
        for (int j = 0; j < M->cols; j++) {
            mat_set(result, j, i, mat_get(M, i, j));
            //result->data[j * result->cols + i] = M->data[i * M->cols + j];
        }
    }

    return result;
}


void mat_copy_data_from_arr(Matrix* dest, float arr[16][5]){
    for (int i = 0; i < dest->rows; i++) {
        for (int j = 0; j < dest->cols; j++) {
            mat_set(dest, i, j, arr[i][j]);
            //dest->data[j * result->cols + i] = M->data[i * M->cols + j];
        }
    }
}

void mat_random_fill(Matrix* m, float max){
    for(int i = 0; i < (m->rows*m->cols); i++){
        m->data[i] = random_float(max);
    }
}

void mat_apply_function(Matrix* M, float (*function)(float)) {
    for (int i = 0; i < M->rows; i++) {
        for (int j = 0; j < M->cols; j++) {
            float value = mat_get(M, i, j);
            mat_set(M, i, j, function(value));
        }
    }
}

