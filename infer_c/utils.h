/* utils.h – public helpers shared across the C runtime */

#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include <stddef.h>
#include "tensor.h"

/* Min/max value from INT8 buffer */
int8_t find_min_int8(const int8_t *data, size_t n);
int8_t find_max_int8(const int8_t *data, size_t n);

/* Numeric helpers */
int8_t clamp_int8 (int32_t v);                      /* saturate to -127...+127 */
float  round_float(float   x);                      /* basic rounding */
float  mean_int8  (const int8_t *data, size_t n);
int    get_idx_3d (int c, int h, int w, int H, int W);

/* Quantisation utils */
void quantize_weights (Tensor *src, Tensor *dst,
                       float  *scale, uint8_t keep_float);

void dequantize_weights(const Tensor *q_w,
                        Tensor       *fp_w,
                        float         scale);

/* Tensor ops */
void ensure_fp32(Tensor *t);  /* allocates f_data if missing or aliases q_data */

void apply_attention_gate(Tensor *residual,
                          const ConstTensor *gate,
                          const ConstTensor *scale);

/* Debug/testing */
int compare_tensors(const Tensor *a,
                    const Tensor *b,
                    float         tol);

#endif /* UTILS_H */
