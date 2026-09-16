#ifndef BLOCK_3_H
#define BLOCK_3_H

#include "ap_fixed.h"

typedef ap_fixed<8, 4>   data_t;     // interface externe (in), inchangé — compatible block_2
typedef ap_fixed<16, 4>  fc_act_t;   // au lieu de <16,8>
typedef ap_fixed<26, 16> acc3_t;     // widened: 256-term FC1 dot product can reach ~16384, needs headroom
typedef ap_fixed<8, 4>   weight_t;

#define C2_OUT_CH   16
#define P2_OUT_SIZE 4
#define FLAT_SIZE   (C2_OUT_CH * P2_OUT_SIZE * P2_OUT_SIZE)  // 256

#define FC1_OUT     120
#define FC2_OUT     84
#define FC3_OUT     10

void block_3(
    data_t in[C2_OUT_CH][P2_OUT_SIZE][P2_OUT_SIZE],
    int    &pred_class
);

#endif
