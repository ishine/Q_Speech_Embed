#include<math.h>
#include "softmax.h"
#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>

void softmax(Tensor *x) {
#ifndef QUANT_MODE_QAT_SQ
    // Float mode.
    if (!x->f_data) {
    fprintf(stderr, "softmax: f_data is NULL!\n");
    exit(1);
}
    float max_val = x->f_data[0];
    for (int i = 1; i < x->size; i++) {
        if (x->f_data[i] > max_val) max_val = x->f_data[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < x->size; i++) {
        x->f_data[i] = expf(x->f_data[i] - max_val);  // subtract max for stability.
        sum += x->f_data[i];
    }

    for (int i = 0; i < x->size; i++) {
        x->f_data[i] /= sum;
    }
#else
    // Quantized: placeholder.
    for (int i = 0; i < x->size; i++) {
        x->q_data[i] = x->q_data[i];  // TODO: replace with actual quantized softmax.
    }
#endif
}
