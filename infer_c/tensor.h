/* tensor.h – public tensor struct & helpers
 * Compatible with both FP32 and INT8 payloads.
 */

#ifndef TENSOR_H
#define TENSOR_H

#include "weights_meta.h"
#include <stdint.h>
#include <stddef.h>

/* Runtime limits for scratch tensor size (in elements, not bytes) */
#ifndef TENSOR_SCRATCH_LIMIT
#define TENSOR_SCRATCH_LIMIT   200000
#endif

/* Tensor => mutable, used during inference, owns heap storage */
typedef struct {
    uint8_t  dims;       /* number of dimensions */
    size_t   size;       /* product of shape */
    uint8_t *shape;      /* [dims] */
    float   *f_data;     /* populated if FP32 */
    int8_t  *q_data;     /* populated if INT8 */
} Tensor;

/* ConstTensor => read-only view from exported weights (g_wt_table[]) */
typedef struct {
    uint8_t         dims;
    size_t          size;
    const uint8_t  *shape;
    const int8_t   *q_data;   /* used when type == 0 in WT_Entry */
    const float    *f_data;   /* used when type == 1 */
} ConstTensor;

/* Wrap one WT_Entry into a ConstTensor */
static inline ConstTensor
wrap_const_tensor(const WT_Entry *e)
{
    ConstTensor t = {0};
    t.dims        = (uint8_t)e->dims;
    t.size        = e->size;
    t.shape       = e->shape;
    t.q_data      = (e->type == 0) ? (const int8_t *)e->ptr : NULL;
    t.f_data      = (e->type == 1) ? (const float  *)e->ptr : NULL;
    return t;
}

/* Tensor construction and management */
Tensor  create_tensor (const uint8_t *shape, int dims);
Tensor  clone_tensor  (const Tensor *src);
void    free_tensor   (Tensor *t);

/* Tensor file I/O */
Tensor  f_load_tensor (const char *file, int dims);  /* loads float32 */
Tensor   load_tensor  (const char *file, int dims);  /* loads int8 */
void     save_tensor  (const char *file, const Tensor *t);

/* Shape utilities */
size_t   get_size     (const uint8_t *shape, int dims);
void     print_shape  (const Tensor *t);
void     print_tensor (const Tensor *t);

#endif /* TENSOR_H */
