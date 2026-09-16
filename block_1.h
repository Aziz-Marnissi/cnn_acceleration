#ifndef BLOCK_1_H
#define BLOCK_1_H

#include "ap_fixed.h"

typedef ap_fixed<8, 4>  data_t;
typedef ap_fixed<20, 10> acc_t;
typedef ap_fixed<8, 2>  weight_t;
#define IMG_SIZE    28
#define C1_K        5
#define C1_OUT_CH   6
#define C1_OUT_SIZE 24

#define P1_OUT_SIZE 12

void block_1(
    data_t img[IMG_SIZE][IMG_SIZE],
    data_t out[C1_OUT_CH][P1_OUT_SIZE][P1_OUT_SIZE]
);

#endif
