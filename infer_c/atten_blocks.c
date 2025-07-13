/* atten_blocks.c – Dual-Gate Attention + BN block */

#include "atten_blocks.h"
#include "qparams.h"
#include "weights_meta.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "tensor.h"
#include "utils.h"
#include "pooling.h"
#include "upsampling.h"
#include "conv2d.h"
#include "activation.h"

#ifdef DEBUG
#  define DBG(fmt, ...)  fprintf(stderr, "[DBG] " fmt "\n", ##__VA_ARGS__)
#else
#  define DBG(fmt, ...)  ((void)0)
#endif

/* Element-wise multiplication with float promotion */
static Tensor elem_mul(const Tensor *a, const Tensor *b)
{
    Tensor out = create_tensor(a->shape, a->dims);
    ensure_fp32(&out);

    for (size_t i = 0; i < a->size; ++i)
        out.f_data[i] =
            (a->f_data ? a->f_data[i] : (float)a->q_data[i]) *
            (b->f_data ? b->f_data[i] : (float)b->q_data[i]);

    return out;
}

/* Element-wise addition with float promotion */
static Tensor elem_add(const Tensor *a, const Tensor *b)
{
    Tensor out = create_tensor(a->shape, a->dims);
    ensure_fp32(&out);

    for (size_t i = 0; i < a->size; ++i)
        out.f_data[i] =
            (a->f_data ? a->f_data[i] : (float)a->q_data[i]) +
            (b->f_data ? b->f_data[i] : (float)b->q_data[i]);

    return out;
}

/* Per-channel scaling: y = x * (q * s) */
static Tensor channel_scale(const Tensor *x,
                            const ConstTensor *q,
                            const ConstTensor *s)
{
    int C  = x->shape[0];
    int HW = (int)(x->size / C);

    Tensor out = create_tensor(x->shape, x->dims);
    ensure_fp32(&out);

    int q_scalar = (!q || q->size == 1);
    int s_scalar = (!s || s->size == 1);

    for (int c = 0; c < C; ++c) {
        float k = 1.0f;

        if (q) {
            k *= q->f_data
                  ? q->f_data[q_scalar ? 0 : c]
                  : (float)q->q_data[q_scalar ? 0 : c];
        }
        if (s) {
            k *= s->f_data
                  ? s->f_data[s_scalar ? 0 : c]
                  : (float)s->q_data[s_scalar ? 0 : c];
        }

        for (int i = 0; i < HW; ++i) {
            int idx = c * HW + i;
            float v = x->f_data ? x->f_data[idx] : (float)x->q_data[idx];
            out.f_data[idx] = v * k;
        }
    }
    return out;
}

/* Single residual branch with attention gate and scale */
Tensor GateResidual(const Tensor *x, const char *prefix)
{
    /* 0. Copy input for skip connection */
    Tensor residual = clone_tensor(x);

    /* 1. Downsample with 2×2 max-pool */
    Tensor q = max_pool2d(x, 2, 2);

    /* 2. Grouped 1×1 convolution */
    char name[96];
    snprintf(name, sizeof(name), "%s.group_conv", prefix);
    QParams gp = get_qparams(name);
    Tensor k = conv2d(&q, &gp.w,
                         gp.s.size ? &gp.s : NULL,
                         gp.b.size ? &gp.b : NULL,
                         1, 0);
    relu(&k);
    free_tensor(&q);

    /* 3. Pointwise 1×1 convolution */
    snprintf(name, sizeof(name), "%s.pointwise_conv", prefix);
    QParams pw = get_qparams(name);
    Tensor k2 = conv2d(&k, &pw.w,
                          pw.s.size ? &pw.s : NULL,
                          pw.b.size ? &pw.b : NULL,
                          1, 0);
    relu(&k2);
    free_tensor(&k);

    /* 4. Expansion 1×1 convolution */
    snprintf(name, sizeof(name), "%s.expand_conv", prefix);
    QParams ex = get_qparams(name);
    Tensor a = conv2d(&k2, &ex.w,
                         ex.s.size ? &ex.s : NULL,
                         ex.b.size ? &ex.b : NULL,
                         1, 0);
    relu(&a);
    free_tensor(&k2);

    /* 5. Upsample and apply sigmoid gate */
    Tensor up = upsample_nearest(&a, 2);
    free_tensor(&a);

    Tensor gate = sigmoid((ConstTensor *)&up);
    free_tensor(&up);

    Tensor v = elem_mul(&residual, &gate);
    free_tensor(&gate);

    /* 6. Apply learned per-channel scale */
    snprintf(name, sizeof(name), "%s.scale", prefix);
    QParams sc = get_qparams(name);
    Tensor scaled = channel_scale(&v,
                                  sc.w.size ? &sc.w : NULL,
                                  sc.s.size ? &sc.s : NULL);
    free_tensor(&v);

    /* 7. Add residual back */
    Tensor out = elem_add(&residual, &scaled);
    free_tensor(&scaled);
    free_tensor(&residual);

    return out;
}

/* Dual-gated residual block with batch norm structure */
Tensor Attn_BN_Block(Tensor *x, const char *base)
{
    char sub[64];

    /* First residual branch */
    snprintf(sub, sizeof(sub), "%s.layer1", base);
    Tensor g0 = GateResidual(x, sub);
    relu(&g0);

    /* Second residual branch */
    snprintf(sub, sizeof(sub), "%s.layer3", base);
    Tensor g1 = GateResidual(&g0, sub);
    relu(&g1);
    free_tensor(&g0);

    /* Final skip connection */
    Tensor out = elem_add(&g1, x);
    free_tensor(&g1);

    return out;
}
