/* activation.c – sigmoid and ReLU activation functions (INT8 or FP32 compatible) */

#include "activation.h"
#include "utils.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

/*
 * Applies the sigmoid activation function: sigma(x) = 1 / (1 + exp(-x))
 * Accepts ConstTensor with either float or int8 data.
 * Converts int8 input to float if needed.
 */
Tensor sigmoid(const ConstTensor *in)
{
    if (!in) {
        fputs("sigmoid: NULL input\n", stderr);
        exit(EXIT_FAILURE);
    }

    /* 1. Identify float data source, converting from int8 if necessary */
    const size_t n = in->size;
    const float *src = in->f_data;
    float       *tmp = NULL;

    if (!src) {
        if (!in->q_data) {
            fputs("sigmoid: both f_data and q_data are NULL\n", stderr);
            exit(EXIT_FAILURE);
        }
        tmp = (float *)malloc(n * sizeof(float));
        if (!tmp) {
            perror("sigmoid/tmp");
            exit(EXIT_FAILURE);
        }
        for (size_t i = 0; i < n; ++i)
            tmp[i] = (float)in->q_data[i];
        src = tmp;
    }

    /* 2. Allocate output tensor with same shape and float storage */
    uint8_t shape[4] = {0};
    memcpy(shape, in->shape, in->dims);
    Tensor out = create_tensor(shape, in->dims);
    ensure_fp32(&out);

    /* 3. Apply sigmoid function element-wise */
    for (size_t i = 0; i < n; ++i) {
        const float x = src[i];
        out.f_data[i] = 1.f / (1.f + expf(-x));
    }

    free(tmp);  /* Only free if conversion buffer was allocated */
    return out; /* Caller must free the returned tensor */
}

/*
 * Applies in-place ReLU: replaces negative values with zero.
 * Works for both float and int8 tensors.
 */
void relu(Tensor *t)
{
    if (!t) {
        fprintf(stderr, "relu: NULL tensor\n");
        exit(1);
    }

    if (!t->f_data)
        ensure_fp32(t);

    if (t->f_data) {
        for (size_t i = 0; i < t->size; ++i)
            if (t->f_data[i] < 0.0f)
                t->f_data[i] = 0.0f;
    } else {
        for (size_t i = 0; i < t->size; ++i)
            if (t->q_data[i] < 0)
                t->q_data[i] = 0;
    }
}
