#include "block_1.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>

void block_1(
    data_t img[IMG_SIZE][IMG_SIZE],
    data_t out[C1_OUT_CH][P1_OUT_SIZE][P1_OUT_SIZE]
);

int main() {
    data_t img[IMG_SIZE][IMG_SIZE];
    data_t out[C1_OUT_CH][P1_OUT_SIZE][P1_OUT_SIZE];

    srand(42);
    for (int i = 0; i < IMG_SIZE; i++)
        for (int j = 0; j < IMG_SIZE; j++)
            img[i][j] = (data_t)(((rand() % 2000) / 1000.0f) - 1.0f);

    block_1(img, out);

    printf("block1 output (C1_OUT_CH=%d, P1_OUT_SIZE=%d):\n", C1_OUT_CH, P1_OUT_SIZE);

    int neg_count = 0, total = 0;
    float min_v = 1e9f, max_v = -1e9f, sum = 0.0f;

    for (int c = 0; c < C1_OUT_CH; c++) {
        for (int i = 0; i < P1_OUT_SIZE; i++) {
            for (int j = 0; j < P1_OUT_SIZE; j++) {
                float v = out[c][i][j].to_float();
                total++;
                if (v < 0.0f) neg_count++;
                if (v < min_v) min_v = v;
                if (v > max_v) max_v = v;
                sum += v;
            }
        }
    }

    printf("Total elements : %d\n", total);
    printf("Min value      : %f\n", min_v);
    printf("Max value      : %f\n", max_v);
    printf("Mean value     : %f\n", sum / total);
    printf("Negative count : %d (expected 0, ReLU applied before pooling)\n", neg_count);

    printf("\nChannel 0, row 0:\n");
    for (int j = 0; j < P1_OUT_SIZE; j++) {
        printf("%8.4f ", out[0][0][j].to_float());
    }
    printf("\n");

    int pass = (neg_count == 0);
    printf("\nTEST %s\n", pass ? "PASSED" : "FAILED");
    return pass ? 0 : 1;
}
