#ifndef BLOCK_2_H
#define BLOCK_2_H

#include "ap_fixed.h"

typedef ap_fixed<8, 4>  data_t;
// block_2.h
typedef ap_fixed<24, 14, AP_TRN, AP_SAT> acc2_t;// block_2.h
typedef ap_fixed<8, 4>  weight_t;
#define C1_IN_CH    6
#define C1_IN_SIZE  12

#define C2_K        5
#define C2_OUT_CH   16
#define C2_OUT_SIZE 8

#define P2_OUT_SIZE 4

void block_2(
    data_t in[C1_IN_CH][C1_IN_SIZE][C1_IN_SIZE],
    data_t out[C2_OUT_CH][P2_OUT_SIZE][P2_OUT_SIZE]
);

#endif
