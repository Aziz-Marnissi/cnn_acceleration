#include "block_1.h"
#include "weights_trained.h"

template<typename T>
T relu(T x) {
#pragma HLS INLINE
    return (x < 0) ? (T)0 : x;
}
static void conv1_layer(
    data_t img[IMG_SIZE][IMG_SIZE],
    data_t out[C1_OUT_CH][C1_OUT_SIZE][C1_OUT_SIZE]
) {
#pragma HLS INLINE off

    data_t line_buf[C1_K-1][IMG_SIZE];
#pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1
    data_t window[C1_K][C1_K];
#pragma HLS ARRAY_PARTITION variable=window complete dim=0

    for (int i = 0; i < IMG_SIZE; i++) {
        for (int j = 0; j < IMG_SIZE; j++) {
#pragma HLS PIPELINE II=1
            // shift window columns
            for (int r = 0; r < C1_K; r++) {
#pragma HLS UNROLL
                for (int c = 0; c < C1_K-1; c++) {
#pragma HLS UNROLL
                    window[r][c] = window[r][c+1];
                }
            }
            // shift line buffer rows, load new pixel at bottom
            data_t new_pix = img[i][j];
            for (int r = 0; r < C1_K-1; r++) {
#pragma HLS UNROLL
                window[r][C1_K-1] = line_buf[r][j];
            }
            window[C1_K-1][C1_K-1] = new_pix;
            for (int r = 0; r < C1_K-2; r++) {
#pragma HLS UNROLL
                line_buf[r][j] = line_buf[r+1][j];
            }
            if (C1_K > 1) line_buf[C1_K-2][j] = new_pix;

            if (i >= C1_K-1 && j >= C1_K-1) {
                int oi = i - (C1_K-1);
                int oj = j - (C1_K-1);
                for (int co = 0; co < C1_OUT_CH; co++) {
                    acc_t sum = conv1_b[co];
                    for (int ki = 0; ki < C1_K; ki++)
                        for (int kj = 0; kj < C1_K; kj++) {
#pragma HLS UNROLL
                            sum += (acc_t)window[ki][kj] * (acc_t)conv1_w[co][ki][kj];
                        }
                    out[co][oi][oj] = relu<acc_t>(sum);
                }
            }
        }
    }
}

static void pool1_layer(
    data_t in[C1_OUT_CH][C1_OUT_SIZE][C1_OUT_SIZE],
    data_t out[C1_OUT_CH][P1_OUT_SIZE][P1_OUT_SIZE]
) {
#pragma HLS INLINE off

    for (int c = 0; c < C1_OUT_CH; c++) {
        for (int i = 0; i < P1_OUT_SIZE; i++) {
            for (int j = 0; j < P1_OUT_SIZE; j++) {
#pragma HLS PIPELINE II=1
                data_t a  = in[c][2*i][2*j];
                data_t b  = in[c][2*i][2*j+1];
                data_t c1 = in[c][2*i+1][2*j];
                data_t d  = in[c][2*i+1][2*j+1];
                data_t m1 = (a > b)  ? a  : b;
                data_t m2 = (c1 > d) ? c1 : d;
                out[c][i][j] = (m1 > m2) ? m1 : m2;
            }
        }
    }
}

void block_1(
    data_t img[IMG_SIZE][IMG_SIZE],
    data_t out[C1_OUT_CH][P1_OUT_SIZE][P1_OUT_SIZE]
) {
#pragma HLS INTERFACE m_axi port=img bundle=gmem0 max_read_burst_length=32
#pragma HLS INTERFACE m_axi port=out bundle=gmem1 max_write_burst_length=32
#pragma HLS INTERFACE s_axilite port=return

    data_t img_local[IMG_SIZE][IMG_SIZE];
    data_t c1[C1_OUT_CH][C1_OUT_SIZE][C1_OUT_SIZE];

#pragma HLS BIND_STORAGE variable=img_local type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=c1        type=ram_2p impl=bram
    for (int i = 0; i < IMG_SIZE; i++) {
        for (int j = 0; j < IMG_SIZE; j++) {
    #pragma HLS PIPELINE II=1
    #pragma HLS LOOP_FLATTEN off
            img_local[i][j] = img[i][j];
        }
    }
    conv1_layer(img_local, c1);
    pool1_layer(c1, out);
}
