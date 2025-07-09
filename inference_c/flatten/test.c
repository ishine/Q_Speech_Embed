/*  block1/test.c  – first Attention-BN residual block */
#include <stdio.h>
#include <stdlib.h>

#include "../tensor.h"
#include "../modules.h"     /* Attn_BN_Block */
#include "../utils.h"
#include "../weights.h"
#include "../common.h"

/* helper: ConstTensor cast */
#define CT(id)  ((const ConstTensor*)model_weights[(id)].address)

int main(void)
{
    puts("=== block-1 test ===");

    /* 1. load reference input (Conv-1 output, before pooling) */
    Tensor in = f_load_tensor("../logs/01_conv1_out.bin", 3);
    print_shape(&in);

    /* 2. run the exact same C kernel
           ─ Conv-1 consumed weight-IDs 0..2
           ─ block-1 starts at ID 3                    */
    uint8_t layer_id = 3;          /* keep in sync with tinyspeech.c */
    Tensor out = Attn_BN_Block(&in, &layer_id);

    print_shape(&out);

    /* 3. compare to PyTorch reference */
    Tensor ref = f_load_tensor("../logs/02_block1_out.bin", 3);
    compare_tensors(&out, &ref, 1e-4f);

    /* tidy */
    free_tensor(&in);
    free_tensor(&out);
    free_tensor(&ref);
    return 0;
}
