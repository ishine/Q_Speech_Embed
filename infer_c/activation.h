#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "tensor.h"

/* allocates a brand-new FP32 tensor (caller frees) */
Tensor sigmoid(const ConstTensor *in);

/* in-place ReLU (works on either fp32 or int8 buffers) */
void   relu(Tensor *t);

#endif /* ACTIVATION_H */
