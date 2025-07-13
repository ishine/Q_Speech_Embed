/* upsampling.c – nearest-neighbour upsample (scale >= 1) */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "tensor.h"
#include "utils.h"

Tensor upsample_nearest(const Tensor *in, int scale)
{
    if (scale < 1) {
        fprintf(stderr, "upsample_nearest: scale=%d (must be >=1)\n", scale);
        exit(EXIT_FAILURE);
    }
    if (in->dims != 3 && in->dims != 4) {
        fprintf(stderr, "upsample_nearest: dims=%d (want 3 or 4)\n", in->dims);
        exit(EXIT_FAILURE);
    }

    /* Unpack shape */
    const int N = (in->dims == 4) ? in->shape[0] : 1;
    const int C = (in->dims == 4) ? in->shape[1] : in->shape[0];
    const int H =  in->shape[in->dims - 2];
    const int W =  in->shape[in->dims - 1];

    const int H_out = H * scale;
    const int W_out = W * scale;

    /* Build output shape */
    uint8_t oshape[4];
    int     odims;
    if (in->dims == 4) {
        odims     = 4;
        oshape[0] = (uint8_t)N;
        oshape[1] = (uint8_t)C;
        oshape[2] = (uint8_t)H_out;
        oshape[3] = (uint8_t)W_out;
    } else {
        odims     = 3;
        oshape[0] = (uint8_t)C;
        oshape[1] = (uint8_t)H_out;
        oshape[2] = (uint8_t)W_out;
    }

    Tensor out = create_tensor(oshape, odims);

    /* Ensure both tensors have FP32 data */
    ensure_fp32((Tensor *)in);  /* cast away const => just a read-only promise */
    ensure_fp32(&out);
    free(out.q_data);
    out.q_data = NULL;

    assert(out.f_data && in->f_data);

    /* Nearest-neighbour copy loop */
    for (int n = 0; n < N; ++n)
    for (int c = 0; c < C; ++c)
    for (int oh = 0; oh < H_out; ++oh)
    for (int ow = 0; ow < W_out; ++ow)
    {
        const int ih = oh / scale;
        const int iw = ow / scale;

        const size_t iidx = (in->dims == 4)
                          ? (((n * C + c) * H + ih) * W + iw)
                          :  ((c * H + ih) * W + iw);

        const size_t oidx = (odims == 4)
                          ? (((n * C + c) * H_out + oh) * W_out + ow)
                          :  ((c * H_out + oh) * W_out + ow);

        out.f_data[oidx] = in->f_data[iidx];
    }

    return out;
}
