/* pooling.c – simple max-pool & adaptive-avg-pool helpers
 *
 *   • max_pool2d()          – k×k kernel, stride = s   (dims 3 or 4)
 *   • adaptive_avg_pool2d() – global H×W  →  1×1
 *
 *  All outputs are allocated with create_tensor() and promoted to FP32
 *  via ensure_fp32(), so  out.f_data  is always valid.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"
#include "utils.h" 

/*  adaptive AvgPool2D (global) → output H = W = 1  */

Tensor adaptive_avg_pool2d(Tensor *x)
{
    if (x->dims != 3 && x->dims != 4) {
        fprintf(stderr, "adaptive_avg_pool2d: dims=%d (want 3 or 4)\n",
                x->dims);
        exit(EXIT_FAILURE);
    }

    int N = (x->dims == 4) ? x->shape[0] : 1;
    int C = (x->dims == 4) ? x->shape[1] : x->shape[0];
    int H = x->shape[x->dims - 2];
    int W = x->shape[x->dims - 1];

    /* correct rank? */
    uint8_t oshape[4];
    int     odims;
    if (x->dims == 4) {               /* N,C,1,1 – rank-4 */
        odims = 4;
        oshape[0] = (uint8_t)N;
        oshape[1] = (uint8_t)C;
        oshape[2] = 1;
        oshape[3] = 1;
    } else {                          /* C,1,1  – rank-3  */
        odims = 3;
        oshape[0] = (uint8_t)C;
        oshape[1] = 1;
        oshape[2] = 1;
    }

    Tensor out = create_tensor(oshape, odims);
    ensure_fp32(&out);

    /* averaging loop */
    for (int n = 0; n < N; ++n)
    for (int c = 0; c < C; ++c)
    {
        float sum = 0.0f;

        for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w)
        {
            size_t idx = (x->dims == 4)
                       ? (((n * C + c) * H + h) * W + w)
                       :  ((c * H + h) * W + w);

            sum += x->f_data
                     ? x->f_data[idx]
                     : (x->q_data ? (float)x->q_data[idx] : 0.0f);
        }

        float mean = sum / (float)(H * W);

        size_t oidx = (odims == 4)
                    ? (((n * C + c) * 1 + 0) * 1 + 0)   /* flattened */
                    :  (c * 1 + 0);

        out.f_data[oidx] = mean;
    }
    return out;
}


/*  MaxPool2D (k×k / stride s)  */
Tensor max_pool2d(const Tensor *in, int k, int s)
{
    ensure_fp32((Tensor *)in);   /* guarantee f_data is present */

    if (in->dims != 3 && in->dims != 4) {
        fprintf(stderr, "max_pool2d: dims=%d (want 3 or 4)\n", in->dims);
        exit(EXIT_FAILURE);
    }

    /* unpack shape */
    int N = (in->dims == 4) ? in->shape[0] : 1;
    int C = (in->dims == 4) ? in->shape[1] : in->shape[0];
    int H = in->shape[in->dims - 2];
    int W = in->shape[in->dims - 1];

    int H_out = (H - k) / s + 1;
    int W_out = (W - k) / s + 1;

    /* build correct output shape */
    uint8_t oshape[4];
    int     odims;
    if (in->dims == 4) {               /* N,C,H_out,W_out */
        odims = 4;
        oshape[0] = (uint8_t)N;
        oshape[1] = (uint8_t)C;
        oshape[2] = (uint8_t)H_out;
        oshape[3] = (uint8_t)W_out;
    } else {                           /* C,H_out,W_out */
        odims = 3;
        oshape[0] = (uint8_t)C;
        oshape[1] = (uint8_t)H_out;
        oshape[2] = (uint8_t)W_out;
    }

    Tensor out = create_tensor(oshape, odims);
    ensure_fp32(&out);

    /* main loop */
    for (int n = 0; n < N; ++n)
    for (int c = 0; c < C; ++c)
    for (int oh = 0; oh < H_out; ++oh)
    for (int ow = 0; ow < W_out; ++ow)
    {
        float m = -INFINITY;

        for (int kh = 0; kh < k; ++kh)
        for (int kw = 0; kw < k; ++kw)
        {
            int ih = oh * s + kh;
            int iw = ow * s + kw;

            size_t idx = (in->dims == 4)
                       ? (((n * C + c) * H + ih) * W + iw)
                       :  ((c * H + ih) * W + iw);

            if (in->f_data[idx] > m)
                m = in->f_data[idx];
        }

        size_t oidx = (odims == 4)
                    ? (((n * C + c) * H_out + oh) * W_out + ow)
                    :  ((c * H_out + oh) * W_out + ow);

        out.f_data[oidx] = m;
    }

    return out;
}
