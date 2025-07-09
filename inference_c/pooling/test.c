/*  pooling/test.c  – verify max_pool2d (2× down-sample) */
#include <stdio.h>

#include "../tensor.h"
#include "../pooling.h"
#include "../utils.h"

int main(void)
{
    puts("=== pooling test (kernel=2, stride=2) ===");

    /* 1. load input that Python saved right after Conv-1 + ReLU */
    Tensor in = f_load_tensor("../logs/01_conv1_out.bin", 3);
    puts("Input loaded");
    print_shape(&in);

    /* 2. run the kernel under test */
    Tensor out = max_pool2d(&in, /*kernel=*/2, /*stride=*/2);
    puts("max_pool2d done");
    print_shape(&out);

    /* 3. load Python reference result (see note below) */
    Tensor ref = f_load_tensor("logs/01_conv1_out_pool2.bin", 3);
    puts("Reference loaded");
    print_shape(&ref);

    /* 4. compare – tolerance can be very small (pure FP32) */
    compare_tensors(&out, &ref, /*tolerance=*/1e-5f);

    /* optional: keep a copy */
    save_tensor("pool_out.bin", &out);

    /* 5. tidy-up */
    free_tensor(&in);
    free_tensor(&out);
    free_tensor(&ref);
    return 0;
}
