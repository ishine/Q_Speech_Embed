/* bit_gate_net.c – top-level graph runner for the MCU port */

#include <stdio.h>
#include <stdint.h>

#include "weights_meta.h"
#include "tensor.h"
#include "utils.h"
#include "conv2d.h"
#include "atten_blocks.h"
#include "dense.h"
#include "pooling.h"
#include "softmax.h"
#include "activation.h"
#include "qparams.h"
#include <stddef.h>

#ifdef DEBUG
#  define DBG(fmt, ...)  do { \
        printf("[DBG] " fmt "\n", ##__VA_ARGS__); fflush(stdout); \
    } while (0)
#else
#  define DBG(fmt, ...)  ((void)0)
#endif

/* Helper for printing tensor shape if debugging is enabled */
static void dbg_shape(const char *tag, const Tensor *t)
{
#ifdef DEBUG
    printf("      %-10s ", tag);
    print_shape(t);
    fflush(stdout);
#endif
}

/* Returns NULL if the tensor is empty (used for optional scale/bias) */
static inline const ConstTensor *OPT(const ConstTensor *t)
{
    return (t && t->size) ? t : NULL;
}

/* Finds minimum int8 value in a buffer */
int8_t find_min_int8(const int8_t *data, size_t n)
{
    if (!data || n == 0) return 0;
    int8_t vmin = data[0];
    for (size_t i = 1; i < n; ++i)
        if (data[i] < vmin) vmin = data[i];
    return vmin;
}

/* Finds maximum int8 value in a buffer */
int8_t find_max_int8(const int8_t *data, size_t n)
{
    if (!data || n == 0) return 0;
    int8_t vmax = data[0];
    for (size_t i = 1; i < n; ++i)
        if (data[i] > vmax) vmax = data[i];
    return vmax;
}

/* Full forward pass for the BitGateNet model */
Tensor BitGateNet(Tensor *inp)
{
    DBG("===== BitGateNet()  INPUT =====");
    dbg_shape("inp", inp);

    /* 1. Initial convolution layer */
    QParams q1 = get_qparams("conv1");
    Tensor  x  = conv2d(inp, &q1.w, OPT(&q1.s), OPT(&q1.b), 1, 1);
    printf("conv1 min=%d max=%d\n",
           find_min_int8(x.q_data, x.size),
           find_max_int8(x.q_data, x.size));
    relu(&x);
    dbg_shape("conv1", &x);

    /* 2. Four attention blocks */
    x = Attn_BN_Block(&x, "block1"); dbg_shape("block1", &x);
    x = Attn_BN_Block(&x, "block2"); dbg_shape("block2", &x);
    x = Attn_BN_Block(&x, "block3"); dbg_shape("block3", &x);
    x = Attn_BN_Block(&x, "block4"); dbg_shape("block4", &x);

    /* 3. Final convolution layer */
    QParams q2 = get_qparams("conv2");
    x = conv2d(&x, &q2.w, OPT(&q2.s), OPT(&q2.b), 1, 1);
    relu(&x);
    dbg_shape("conv2", &x);

    /* 4. Global average pooling */
    Tensor pooled = adaptive_avg_pool2d(&x);
    dbg_shape("avgpool", &pooled);
    free_tensor(&x);

    /* 5. Fully connected classification layer */
    QParams qf = get_qparams("fc");
    x = fc_layer(&pooled, &qf.w, OPT(&qf.s), OPT(&qf.b));
    free_tensor(&pooled);
    dbg_shape("fc", &x);

    /* 6. Softmax activation for final prediction */
    softmax(&x);
    dbg_shape("softmax", &x);

    DBG("===== BitGateNet()  DONE  =====");
    return x;  /* caller frees the output */
}
