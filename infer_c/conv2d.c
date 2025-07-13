/* conv2d.c – INT8 weights + optional per-channel or per-tensor scale and bias
 *
 * Fully-patched version, July 2025
 * - Fixes double-index bug (ensures single checked write)
 * - Enforces 255-shape guard for uint8_t-limited runtime
 * - Supports multiple scale layouts in a unified loop
 * - Avoids reading uninitialized data
 */

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "tensor.h"
#include "conv2d.h"
#include "utils.h"

#ifdef DEBUG
#   define DBG(fmt, ...)  fprintf(stderr,"[DBG] conv2d: " fmt "\n", ##__VA_ARGS__)
#   define ERR(fmt, ...)  fprintf(stderr,"\033[31m[ERR] conv2d: " fmt "\033[0m\n",##__VA_ARGS__)
#else
#   define DBG(fmt, ...)  ((void)0)
#   define ERR(fmt, ...)  ((void)0)
#endif
#define DIE(...)  do{ ERR(__VA_ARGS__); abort(); }while(0)

/* Safe access helper for scale or bias values */
static inline float f_at(const ConstTensor *t, size_t i)
{
    if (!t)        return 1.0f;
    if (t->f_data) return t->f_data[i];
    return (float)t->q_data[i];
}

Tensor conv2d(Tensor            *x,
              const ConstTensor *w,
              const ConstTensor *s,    /* optional scale */
              const ConstTensor *b,    /* optional bias */
              int stride, int pad)
{
    /* Input/output geometry */
    const int Cin  = x->shape[0];
    const int Hin  = x->shape[1];
    const int Win  = x->shape[2];
    const int Cout = w->shape[0];
    const int Kh   = w->shape[2];
    const int Kw   = w->shape[3];

    const int Hout = (Hin + 2 * pad - Kh) / stride + 1;
    const int Wout = (Win + 2 * pad - Kw) / stride + 1;

    if (Cout > 255 || Hout > 255 || Wout > 255)
        DIE("tensor dims exceed 255 (Cout=%d Hout=%d Wout=%d)", Cout, Hout, Wout);

    uint8_t oshape[3] = { (uint8_t)Cout, (uint8_t)Hout, (uint8_t)Wout };
    Tensor out = create_tensor(oshape, 3);  /* q_data only, no float output */

    const size_t S = s ? s->size : 0;

    DBG("IN C=%d H=%d W=%d | OUT C=%d K=%dx%d stride=%d pad=%d  (scale=%zu)",
         Cin, Hin, Win, Cout, Kh, Kw, stride, pad, S);

    for (int oc = 0; oc < Cout; ++oc)
    {
        /* Determine scale mode */
        enum { SCALAR, BY_OC, BY_GROUP, SLICED, BY_IC, MATRIX } mode = SCALAR;
        size_t group_sz = 0, slice_step = 0;

        if (S == 0 || S == 1) mode = SCALAR;
        else if (S == (size_t)Cout) mode = BY_OC;
        else if (Cout % S == 0) {
            mode = BY_GROUP;
            group_sz = Cout / S;
        }
        else if (S > (size_t)Cout && S % Cout == 0) {
            mode = SLICED;
            slice_step = S / Cout;
        }
        else if (S == (size_t)Cin) mode = BY_IC;
        else if (S == (size_t)Cout * Cin) mode = MATRIX;
        else mode = SCALAR;

        if (mode == BY_GROUP && Cout % S != 0)
            DIE("scale layout BY_GROUP but Cout=%d not divisible by S=%zu", Cout, S);

        /* Default scale for this output channel */
        float k_oc = 1.0f;
        if      (mode == BY_OC)    k_oc = f_at(s, oc);
        else if (mode == BY_GROUP) k_oc = f_at(s, oc / group_sz);
        else if (mode == SLICED)   k_oc = f_at(s, oc * slice_step);
        else if (mode == SCALAR)   k_oc = f_at(s, 0);

        DBG("  oc=%d scale=%g (mode=%d)", oc, k_oc, mode);

        for (int oh = 0; oh < Hout; ++oh)
        for (int ow = 0; ow < Wout; ++ow)
        {
            int32_t acc = 0;

            for (int ic = 0; ic < Cin; ++ic)
            for (int kh = 0; kh < Kh;  ++kh)
            for (int kw = 0; kw < Kw;  ++kw)
            {
                const int ih = oh * stride - pad + kh;
                const int iw = ow * stride - pad + kw;
                if ((unsigned)ih >= (unsigned)Hin ||
                    (unsigned)iw >= (unsigned)Win)
                    continue;

                const int in_idx = get_idx_3d(ic, ih, iw, Hin, Win);
                const int w_idx  = ((oc * Cin + ic) * Kh + kh) * Kw + kw;

                float k = k_oc;
                if (mode == BY_IC)  k = f_at(s, ic);
                if (mode == MATRIX) k = f_at(s, (size_t)oc * Cin + ic);

                acc += (int32_t)x->q_data[in_idx] * (int32_t)w->q_data[w_idx];

                /* Consume scale early to avoid int32 overflow */
                if (mode == BY_IC || mode == MATRIX) {
                    acc = (int32_t)roundf(acc * k);
                    k = 0.f;  /* scale already applied */
                }
            }

            /* Final scale and bias application */
            float k_final = (mode == BY_IC || mode == MATRIX) ? 1.f : k_oc;
            if (b) acc += (int32_t)roundf(b->f_data[oc]);
            const int8_t v = clamp_int8((int32_t)roundf(acc * k_final));

            const int outidx = get_idx_3d(oc, oh, ow, Hout, Wout);
            if ((size_t)outidx >= out.size)
                DIE("out-of-range write (idx=%d size=%zu)", outidx, out.size);

            out.q_data[outidx] = v;

#ifdef DEBUG
            if (oc == 0 && oh == 0 && ow < 4)
                DBG("      out(0,%d) = %d", ow, v);
#endif
        }
    }

    DBG("Cout=%d Hout=%d Wout=%d out.size=%zu", Cout, Hout, Wout, out.size);
    return out;
}
