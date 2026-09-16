#include "block_3.h"
#include <cstdio>
#include <cstdlib>

int main() {
    data_t in[C2_OUT_CH][P2_OUT_SIZE][P2_OUT_SIZE];
    int pred_class;

    srand(42);
    for (int c = 0; c < C2_OUT_CH; c++)
        for (int i = 0; i < P2_OUT_SIZE; i++)
            for (int j = 0; j < P2_OUT_SIZE; j++)
                in[c][i][j] = (data_t)(((rand() % 2000) / 1000.0f) - 1.0f);

    block_3(in, pred_class);

    printf("block3 output:\n");
    printf("Predicted class : %d\n", pred_class);

    int pass = (pred_class >= 0 && pred_class < FC3_OUT);
    printf("\nTEST %s\n", pass ? "PASSED" : "FAILED");
    return pass ? 0 : 1;
}
