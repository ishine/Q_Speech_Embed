/* qparams.c – tolerant get_qparams() */

#include "qparams.h"
#include "weights_meta.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

/* reusable 1-element tensor filled with 1.0 */
static float one_val = 1.0f;
static ConstTensor ONE = {
    .dims  = 1,
    .size  = 1,
    .shape = (uint8_t[1]){1},
    .q_data = NULL,
    .f_data = &one_val
};

QParams get_qparams(const char *prefix)
{
    char buf[96];
    bool is_scale_node = (strstr(prefix, ".scale") != NULL);

    /* 1. weight (required, unless this is a *.scale node) */
    snprintf(buf, sizeof(buf), "%s.weight", prefix);
    const WT_Entry *eW = find_named_entry(buf);

    if (!eW && !is_scale_node) {              /* real layer ⇒ fatal      */
        fprintf(stderr,
                "get_qparams: required blob '%s' missing!\n", buf);
        exit(EXIT_FAILURE);
    }
    ConstTensor W = eW ? wrap_const_tensor(eW) : ONE;   /* default = 1 */

    /* 2. scale (opt) */
    snprintf(buf, sizeof(buf), "%s.scale", prefix);
    const WT_Entry *eS = find_named_entry(buf);
    ConstTensor S = eS ? wrap_const_tensor(eS) : ONE;   /* default = 1 */

    /* 3. bias (opt) */
    snprintf(buf, sizeof(buf), "%s.bias", prefix);
    const WT_Entry *eB = find_named_entry(buf);
    ConstTensor B = eB ? wrap_const_tensor(eB) : (ConstTensor){0};

    return (QParams){ .w = W, .s = S, .b = B };
}
