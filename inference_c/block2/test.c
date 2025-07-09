/* block2/test.c – verify Attention-BN Block-2 */
#include <stdio.h>
#include <stdlib.h>

#include "../tensor.h"
#include "../modules.h"
#include "../utils.h"
#include "../weights.h"
#include "../common.h"

int main(void)
{
    setvbuf(stdout, NULL, _IOLBF, 0);   /* line-buffer */

    puts("=== block-2 test ===");

    Tensor in  = f_load_tensor("logs/02_block1_out.bin", 3);
    Tensor ref = f_load_tensor("logs/03_block2_out.bin", 3);

    if (!in.f_data || !ref.f_data) {
        fputs("ERROR: could not open .bin files\n", stderr);
        return 1;
    }

    uint8_t layer_id = ID_BLOCK2_LAYER1_GROUP_CONV_WEIGHT_Q;
    Tensor out = Attn_BN_Block(&in, &layer_id);

    print_shape(&out);
    int ok = compare_tensors(&out, &ref, 1e-3f);
    puts(ok ? "PASS" : "FAIL");

    free_tensor(&in); free_tensor(&ref); free_tensor(&out);
    return ok ? 0 : 1;
}
