/* dense.c – INT8-weight × FP32-input fully-connected (GEMV) */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "dense.h"
#include "utils.h"

/* Safe scalar read for 1D bias or scale tensors */
static inline float VAL_1D(const ConstTensor *t, int i)
{
    return t->f_data ? t->f_data[i] : (float)t->q_data[i];
}

Tensor fc_layer(const Tensor      *x_raw,
                const ConstTensor *w_q,
                const ConstTensor *scale,
                const ConstTensor *bias)
{
    /* 1. Flatten if input is C×1×1 (output of global pooling) */
    Tensor  x_tmp;  /* temp storage for flattened input if needed */
    Tensor *x = (Tensor *)x_raw;

    if (x_raw->dims == 3 && x_raw->shape[1] == 1 && x_raw->shape[2] == 1) {
        uint8_t flat_shape[1] = { x_raw->shape[0] };
        x_tmp = create_tensor(flat_shape, 1);

        ensure_fp32(&x_tmp);
        free(x_tmp.q_data);
        x_tmp.q_data = NULL;

        memcpy(x_tmp.f_data,
               x_raw->f_data,
               x_raw->size * sizeof(float));

        x = &x_tmp;
    } else {
        ensure_fp32(x);
    }

    /* 2. Validate input and weight dimensions */
    if (x->dims != 1 || w_q->dims != 2) {
        fprintf(stderr,
                "fc_layer: expected x dims=1 & w dims=2 (got %d / %d)\n",
                x->dims, w_q->dims);
        exit(EXIT_FAILURE);
    }

    const int in_features  = x->shape[0];
    const int out_features = w_q->shape[0];

    /* 3. Allocate output tensor (float) */
    uint8_t oshape[1] = { (uint8_t)out_features };
    Tensor  out = create_tensor(oshape, 1);
    ensure_fp32(&out);

    const int scale_is_scalar = (!scale || scale->size == 1);

    /* 4. Matrix-vector multiply: output[o] = dot(x, W[o]) * scale + bias */
    for (int o = 0; o < out_features; ++o) {
        float acc = 0.0f;

        for (int i = 0; i < in_features; ++i) {
            const int w_idx = o * in_features + i;
            acc += x->f_data[i] * (float)w_q->q_data[w_idx];
        }

        float s = scale_is_scalar ? (scale ? scale->f_data[0] : 1.0f)
                                  : scale->f_data[o];
        acc *= s;

        if (bias && bias->size > o)
            acc += VAL_1D(bias, o);

        out.f_data[o] = acc;
    }

    /* 5. Free temporary buffer if flattening was used */
    if (x == &x_tmp)
        free_tensor(&x_tmp);

    return out;
}
