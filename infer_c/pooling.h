#ifndef POOLING_H
#define POOLING_H
#include "tensor.h"

Tensor max_pool2d(const Tensor *input, int kernel, int stride);
Tensor adaptive_avg_pool2d(Tensor *x);

#endif