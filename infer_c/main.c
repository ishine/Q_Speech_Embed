#include <stdio.h>
#include <stdint.h>
#include "tensor.h"
#include "bit_gate_net.h"
#include "utils.h"
#include "activation.h"
#include "weights_meta.h"




int main(void)
{
    printf("main: start\n");

    /* 1. build input tensor correctly */
    uint8_t shp[3] = { 1, 12, 94 };
    Tensor inp = create_tensor(shp, 3);

    printf("main: input tensor created (size=%zu)\n", inp.size);

    /* 2. fill it with something */
    for (size_t i = 0; i < inp.size; ++i)
        inp.q_data[i] = (int8_t)((i * 13) & 0x7F);

    printf("main: input filled\n");

    /* 3. run the network*/
    Tensor out = BitGateNet(&inp);
    printf("probabilities: ");
    for (size_t i = 0; i < out.size; ++i)
        printf("%f ", out.f_data[i]);
    putchar('\n');
    /* 4. tidy up */
    free_tensor(&out);
    free_tensor(&inp);
    return 0;
}
