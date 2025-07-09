/*  conv2d/test.c  – self-check for the first convolution layer  */
#include <stdio.h>
#include <stdlib.h>

#include "../tensor.h"
#include "../conv2d.h"
#include "../weights.h"
#include "../utils.h"
#include "../common.h"

int main(void)
{
    puts("=== conv2d/test ===");

    /* ── 1. load refe
    rence input ─────────────────────────── */
    Tensor input = f_load_tensor("logs/00_input.bin", 3);
    puts("Input loaded");
    print_tensor(&input);

    /* ── 2. look-up model tensors ────────────────────────── */
    const ConstTensor *weights =
        (const ConstTensor *)model_weights[ID_CONV1_WEIGHT_Q].address;
    const ConstTensor *scale   =
        (const ConstTensor *)model_weights[ID_CONV1_WEIGHT_SCALE].address;

/* Conv-1 in TinySpeech is bias-free, but keep code robust
   in case a future checkpoint _does_ include one. */
#ifdef ID_CONV1_BIAS
    const ConstTensor *bias =
        (const ConstTensor *)model_weights[ID_CONV1_BIAS].address;
#else
    const ConstTensor *bias = NULL;
#endif

    printf("Weights shape : [%u, %u, %u, %u]\n",
           weights->shape[0], weights->shape[1],
           weights->shape[2], weights->shape[3]);
    printf("Scale shape   : [%u]\n", scale->shape[0]);
    if (bias)
        printf("Bias shape    : [%u]\n", bias->shape[0]);
    else
        puts("Bias          : (none)");

    /* quick peek at raw data */
    printf("First 5 weights (q): ");
    for (int i = 0; i < 5; ++i) printf("%d ", weights->q_data[i]);
    printf("\nFirst 5 scales      : ");
    for (int i = 0; i < 5 && i < scale->size; ++i)
        printf("%.6f ", scale->f_data[i]);
    puts("");

    /* ── 3. run conv2d + ReLU (to match Python log) ──────── */
    Tensor output = conv2d(&input, weights, scale, bias,
                           /*stride=*/1, /*pad=*/1);

    relu(&output);                      /* ← Python saved AFTER ReLU */
    puts("Conv2d + ReLU complete");

    /* optionally write the tensor for manual inspection */
    save_tensor("conv2d_out.bin", &output);

    /* ── 4. compare with reference ───────────────────────── */
    Tensor ref = f_load_tensor("logs/01_conv1_out.bin", 3);
    puts("Reference loaded");

    print_shape(&output);
    print_shape(&ref);

    compare_tensors(&output, &ref, /*max_err=*/1.0f);

    /* ── 5. tidy up ──────────────────────────────────────── */
    free_tensor(&input);
    free_tensor(&output);
    free_tensor(&ref);

    return 0;
}
