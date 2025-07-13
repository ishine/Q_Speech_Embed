#ifndef CONV2D_H
#define CONV2D_H

#include "tensor.h"

Tensor conv2d(Tensor*input,
              const ConstTensor *weight,  // Convolution filter kernel (shape: [out_ch, in_ch, kH, kW]).
              const ConstTensor *scale,   // post-conv quant scale (one per output channel).
              const ConstTensor *bias,    // to accumulator before scaling.
              int stride, int pad);
#endif