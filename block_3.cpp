#include "block_3.h"
#include "weights3_trained.h"

template<typename T>
T relu3(T x) {
#pragma HLS INLINE
    return (x < 0) ? (T)0 : x;
}

static void fc1_layer(
    data_t in[FLAT_SIZE],
    fc_act_t out[FC1_OUT]
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=in cyclic factor=8 dim=1
    for (int o = 0; o < FC1_OUT; o++) {
        acc3_t acc[8] = {0,0,0,0,0,0,0,0};
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
        for (int i = 0; i < FLAT_SIZE; i += 8) {
#pragma HLS PIPELINE II=1
            for (int k = 0; k < 8; k++) {
#pragma HLS UNROLL
                acc[k] += (acc3_t)in[i+k] * (acc3_t)fc1_w[o][i+k];
            }
        }
        acc3_t sum = fc1_b[o];
        for (int k = 0; k < 8; k++) {
#pragma HLS UNROLL
            sum += acc[k];
        }
        out[o] = relu3<acc3_t>(sum);
    }
}

static void fc2_layer(
    fc_act_t in[FC1_OUT],
    fc_act_t out[FC2_OUT]
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=in cyclic factor=20 dim=1
    const int GROUPS = 6;     // 120 / 20
    const int GSIZE  = 20;
    for (int o = 0; o < FC2_OUT; o++) {
#pragma HLS PIPELINE II=1
        acc3_t group_sum[GROUPS];
#pragma HLS ARRAY_PARTITION variable=group_sum complete dim=1
        for (int g = 0; g < GROUPS; g++) {
#pragma HLS UNROLL
            acc3_t psum = 0;
            for (int k = 0; k < GSIZE; k++) {
#pragma HLS UNROLL
                int i = g * GSIZE + k;
                psum += (acc3_t)in[i] * (acc3_t)fc2_w[o][i];
            }
            group_sum[g] = psum;
        }
        acc3_t sum = fc2_b[o];
        for (int g = 0; g < GROUPS; g++) {
#pragma HLS UNROLL
            sum += group_sum[g];
        }
        out[o] = relu3<acc3_t>(sum);
    }
}

static void fc3_layer(
    fc_act_t in[FC2_OUT],
    acc3_t out[FC3_OUT]
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=in complete dim=1
    const int GROUPS = 7;     // 84 / 12
    const int GSIZE  = 12;
    for (int o = 0; o < FC3_OUT; o++) {
#pragma HLS PIPELINE II=1
        acc3_t group_sum[GROUPS];
#pragma HLS ARRAY_PARTITION variable=group_sum complete dim=1
        for (int g = 0; g < GROUPS; g++) {
#pragma HLS UNROLL
            acc3_t psum = 0;
            for (int k = 0; k < GSIZE; k++) {
#pragma HLS UNROLL
                int i = g * GSIZE + k;
                psum += (acc3_t)in[i] * (acc3_t)fc3_w[o][i];
            }
            group_sum[g] = psum;
        }
        acc3_t sum = fc3_b[o];
        for (int g = 0; g < GROUPS; g++) {
#pragma HLS UNROLL
            sum += group_sum[g];
        }
        out[o] = sum;  // no ReLU, raw logits into argmax
    }
}

static void argmax_layer(
    acc3_t in[FC3_OUT],
    int &pred_class
) {
#pragma HLS INLINE off
    acc3_t max_val = in[0];
    int max_idx = 0;
    for (int i = 1; i < FC3_OUT; i++) {
#pragma HLS PIPELINE II=1
        if (in[i] > max_val) {
            max_val = in[i];
            max_idx = i;
        }
    }
    pred_class = max_idx;
}

void block_3(
    data_t in[C2_OUT_CH][P2_OUT_SIZE][P2_OUT_SIZE],
    int    &pred_class
) {
#pragma HLS INTERFACE m_axi port=in bundle=gmem0 max_read_burst_length=32
#pragma HLS INTERFACE s_axilite port=pred_class
#pragma HLS INTERFACE s_axilite port=return

    data_t   flat[FLAT_SIZE];
    fc_act_t fc1_out[FC1_OUT];
    fc_act_t fc2_out[FC2_OUT];
    acc3_t   fc3_out[FC3_OUT];

#pragma HLS ARRAY_PARTITION variable=fc1_w cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=fc2_w cyclic factor=20 dim=2
#pragma HLS ARRAY_PARTITION variable=fc3_w complete dim=2

    int idx = 0;
    for (int c = 0; c < C2_OUT_CH; c++) {
        for (int i = 0; i < P2_OUT_SIZE; i++) {
            for (int j = 0; j < P2_OUT_SIZE; j++) {
#pragma HLS PIPELINE II=1
                flat[idx] = in[c][i][j];
                idx++;
            }
        }
    }

    fc1_layer(flat, fc1_out);
    fc2_layer(fc1_out, fc2_out);
    fc3_layer(fc2_out, fc3_out);
    argmax_layer(fc3_out, pred_class);
}
