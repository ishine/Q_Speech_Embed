/* dense.h – fully-connected layer: INT8 weights x FP32 input */

#pragma once
#include "tensor.h"

/*
 * Computes a dense (fully-connected) layer.
 *
 * Inputs:
 *   x     => 1D FP32 input vector [in_features]
 *   w     => 2D INT8 (or FP32) weight matrix [out_features x in_features]
 *   scale => optional FP32 scale (scalar or vector)
 *   bias  => optional FP32 bias vector [out_features]
 *
 * Returns:
 *   FP32 tensor of shape [out_features]
 */
Tensor fc_layer(const Tensor        *x,
                const ConstTensor   *w,
                const ConstTensor   *scale,
                const ConstTensor   *bias);
