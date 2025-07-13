/* qparams.h – tiny wrapper describing one Conv2d’s parameters */
#pragma once
#include <stdint.h>
#include "tensor.h"

/* keep POD / packed – no hidden padding on MCUs */
typedef struct {
    ConstTensor w;           /* int8 weight  kernel  (dims 4 */
    ConstTensor s;           /* fp32 scale  (scalar or per-OC vector) */
    ConstTensor b;           /* fp32 bias   (may be empty) */

    /* cached geometry to spare many shape[] reads */
    uint16_t oc, ic;
    uint8_t  kh, kw;
} QParams;

/* Get all three blobs for a module prefix: e.g. "block1.layer1.group_conv" */
QParams get_qparams(const char *prefix);
