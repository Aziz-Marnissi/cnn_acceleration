#include "block_2.h"
#include "weights2_trained.h"

template<typename T>
T relu2(T x) {
#pragma HLS INLINE
    return (x < 0) ? (T)0 : x;
}

void block_2(
    data_t in[C1_IN_CH][C1_IN_SIZE][C1_IN_SIZE],
    data_t out[C2_OUT_CH][P2_OUT_SIZE][P2_OUT_SIZE]
) {
#pragma HLS INTERFACE m_axi port=in  bundle=gmem0 depth=864
#pragma HLS INTERFACE m_axi port=out bundle=gmem1 max_write_burst_length=8 depth=256
#pragma HLS INTERFACE s_axilite port=in  bundle=CTRL
#pragma HLS INTERFACE s_axilite port=out bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
#pragma HLS ARRAY_PARTITION variable=conv2_w complete dim=2
#pragma HLS ARRAY_PARTITION variable=conv2_w complete dim=3
#pragma HLS ARRAY_PARTITION variable=conv2_w complete dim=4

    // --- Copie locale de l'entree (BRAM, partitionnee pour acces paralleles) ---
    data_t in_local[C1_IN_CH][C1_IN_SIZE][C1_IN_SIZE];
#pragma HLS ARRAY_PARTITION variable=in_local complete dim=1
#pragma HLS ARRAY_PARTITION variable=in_local complete dim=2
#pragma HLS ARRAY_PARTITION variable=in_local complete dim=3

    COPY_IN: for (int c = 0; c < C1_IN_CH; c++) {
        for (int i = 0; i < C1_IN_SIZE; i++) {
            for (int j = 0; j < C1_IN_SIZE; j++) {
#pragma HLS PIPELINE II=1
                in_local[c][i][j] = in[c][i][j];
            }
        }
    }

    // --- Conv2 + ReLU (resultat garde en interne, pas de tableau intermediaire global) ---
    data_t conv_out[C2_OUT_CH][C2_OUT_SIZE][C2_OUT_SIZE];
#pragma HLS BIND_STORAGE variable=conv_out type=ram_2p impl=bram

    CONV2: for (int co = 0; co < C2_OUT_CH; co++) {
        for (int i = 0; i < C2_OUT_SIZE; i++) {
            for (int j = 0; j < C2_OUT_SIZE; j++) {
#pragma HLS PIPELINE II=1
                acc2_t sum = conv2_b[co];
                for (int ci = 0; ci < C1_IN_CH; ci++) {
                    for (int ki = 0; ki < C2_K; ki++) {
                        for (int kj = 0; kj < C2_K; kj++) {
#pragma HLS UNROLL
                            sum += (acc2_t)in_local[ci][i + ki][j + kj] * (acc2_t)conv2_w[co][ci][ki][kj];
                        }
                    }
                }
                conv_out[co][i][j] = relu2<acc2_t>(sum);
            }
        }
    }

    // --- MaxPool 2x2 + ecriture directe vers out (AXI) ---
    POOL2: for (int c = 0; c < C2_OUT_CH; c++) {
        for (int i = 0; i < P2_OUT_SIZE; i++) {
            for (int j = 0; j < P2_OUT_SIZE; j++) {
#pragma HLS PIPELINE II=1
                data_t a  = conv_out[c][2 * i][2 * j];
                data_t b  = conv_out[c][2 * i][2 * j + 1];
                data_t c1 = conv_out[c][2 * i + 1][2 * j];
                data_t d  = conv_out[c][2 * i + 1][2 * j + 1];
                data_t m1 = (a > b) ? a : b;
                data_t m2 = (c1 > d) ? c1 : d;
                out[c][i][j] = (m1 > m2) ? m1 : m2;
            }
        }
    }
}
