#include <stdio.h>
#include <stdlib.h>

float random_float(float max){
    return ((float)rand() / (float)RAND_MAX) * max;
}

float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}