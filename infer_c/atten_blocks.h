/* atten_blocks.h */
#pragma once
#include "tensor.h"

/* no counters */
Tensor GateResidual (const Tensor *x, const char *prefix);
Tensor Attn_BN_Block(Tensor *x,   const char *base); 
