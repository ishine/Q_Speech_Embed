/* utils.c – misc helpers & lightweight numerics */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>

#include "tensor.h"
#include "utils.h"
#include "weights_meta.h"

/* 1. Lookup weight by name (linear search) */
const WT_Entry *find_named_entry(const char *name)
{
    for (uint32_t i = 0; i < g_wt_count; ++i)
        if (strcmp(g_wt_table[i].name, name) == 0)
            return &g_wt_table[i];
    return NULL;
}

/* 2. Small numeric helpers */
int8_t clamp_int8(int32_t v)         { return (v < -127) ? -127 : (v > 127) ? 127 : (int8_t)v; }
float  round_float(float  x)         { return floorf(x + 0.5f); }

float mean_int8(const int8_t *d, size_t n)
{
    int32_t s = 0;
    for (size_t i = 0; i < n; ++i) s += d[i];
    return (float)s / (float)n;
}

int get_idx_3d(int c, int h, int w, int H, int W)
{
    return (c * H + h) * W + w;
}

/* 3. ensure_fp32 => guarantee/refresh valid fp32 view */
#define FP32_PAD 16

void ensure_fp32(Tensor *t)
{
    if (!t) return;

    const size_t need  = (size_t)t->size * sizeof(float) + FP32_PAD;
    const int    alias = (t->f_data == (float *)t->q_data);

    if (!t->f_data || alias) {
        float *buf = (float *)malloc(need);
        if (!buf) { perror("ensure_fp32"); exit(EXIT_FAILURE); }

        if (t->q_data) {
            for (size_t i = 0; i < t->size; ++i)
                buf[i] = (float)t->q_data[i];
        } else if (t->f_data && !alias) {
            memcpy(buf, t->f_data, t->size * sizeof(float));
        }

        t->f_data = buf;

        if (alias) {
            free(t->q_data);
            t->q_data = NULL;
        }
        return;
    }

    float *tmp = (float *)realloc(t->f_data, need);
    if (!tmp) { perror("ensure_fp32/realloc"); exit(EXIT_FAILURE); }
    t->f_data = tmp;
}

/* 4. Quantization helpers */
void quantize_weights(Tensor *src, Tensor *dst,
                      float *scale, uint8_t keep_float)
{
    for (size_t i = 0; i < dst->size; ++i) {
        int32_t q = clamp_int8((int32_t)roundf(src->f_data[i] / *scale));
        if (keep_float) dst->f_data[i] = (float)q;
        else            dst->q_data[i] = (int8_t)q;
    }
}

void dequantize_weights(const Tensor *q_w,
                        Tensor       *fp_w,
                        float         scale)
{
    for (size_t i = 0; i < fp_w->size; ++i)
        fp_w->f_data[i] = (float)q_w->q_data[i] * scale;
}

/* 5. Attention gate application */
static inline float VAL_1D_SAFE(const ConstTensor *t, size_t i)
{
    size_t idx = (t->size == 1) ? 0 : i;
    return t->f_data ? t->f_data[idx] : (float)t->q_data[idx];
}

void apply_attention_gate(Tensor *res,
                          const ConstTensor *g,
                          const ConstTensor *s)
{
    ensure_fp32(res);
    for (size_t i = 0; i < res->size; ++i) {
        float r = res->f_data[i];
        res->f_data[i] = r + r * VAL_1D_SAFE(s, i) * VAL_1D_SAFE(g, i);
    }
}

/* 6. Comparison helper (tiny test) */
int compare_tensors(const Tensor *a, const Tensor *b, float tol)
{
    if (!a || !b || !a->f_data || !b->f_data) {
        puts("compare_tensors: NULL pointer!");
        return 0;
    }
    if (a->size != b->size) {
        printf("Size mismatch: %zu vs %zu\n", a->size, b->size);
        return 0;
    }

    size_t bad = 0;
    for (size_t i = 0; i < a->size; ++i) {
        float diff = fabsf(a->f_data[i] - b->f_data[i]);
        if (diff > tol) {
            if (bad < 10)
                printf("Δ@%zu: %.6f vs %.6f |%.6f|\n",
                       i, a->f_data[i], b->f_data[i], diff);
            ++bad;
        }
    }

    if (bad == 0)
        printf("tensors match (tol <= %.6f)\n", tol);
    else
        printf("%zu / %zu elements differ > %.6f\n", bad, a->size, tol);

    return bad == 0;
}
