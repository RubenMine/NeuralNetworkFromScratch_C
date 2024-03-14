#ifndef MAT_H
#define MAT_H

typedef struct Matrix{
    int rows;
    int cols;
    float* data;
    int offset;
}Matrix;

Matrix* mat_create(int n, int m);
void mat_free(Matrix* mat);

float mat_get(Matrix* M, int row, int col);
void mat_set(Matrix* M, int row, int col, float value);
void mat_print(Matrix* M);

void mat_sum(Matrix* M1, Matrix* M2);
void mat_sub(Matrix* M1, Matrix* M2);
void mat_scalar_mult(Matrix* M, float scalar);
void mat_dot(Matrix* M1, Matrix* M2, Matrix* res);
Matrix* mat_transpose(Matrix* M);

Matrix* mat_extract_submat(Matrix* M, int rows, int cols, int start_row, int start_col);
Matrix* mat_extract_row(Matrix* M, int row);
Matrix* mat_extract_col(Matrix* M, int col);

void mat_copy_data_from_arr(Matrix* dest, float arr[16][5]);
void mat_random_fill(Matrix* m, float max);
void mat_apply_function(Matrix* M, float (*function)(float));
#endif // MAT_H