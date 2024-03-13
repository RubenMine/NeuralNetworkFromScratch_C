#include "mat.c"

float random_float(float max){
    return ((float)rand() / (float)RAND_MAX) * max;
}

void fillWithRandomValue(Matrix* m, float max){
    for(int i = 0; i < (m->cols+m->cols); i++){
        m->data[i] = random_float(max);
    }
    //mat_print(m);
}