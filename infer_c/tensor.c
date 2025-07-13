/* tensor.c – basic tensor helpers
 *   [uint8 dims][int8 / float ...]  => exported weights blobs
 */

#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "utils.h"

/* Compute product of shape dimensions */
size_t get_size(const uint8_t *shape, int dims)
{
    size_t prod = 1;
    for (int i = 0; i < dims; ++i) prod *= shape[i];
    return prod;
}

/* Helper for element count (alias to get_size) */
static size_t elem_count(const uint8_t *shape, int dims)
{
    size_t n = 1;
    for (int i = 0; i < dims; ++i) n *= shape[i];
    return n;
}

/* Create tensor: allocates shape + INT8 payload (q_data), float stays NULL */
Tensor create_tensor(const uint8_t *shape, int dims)
{
    const size_t n_elem = elem_count(shape, dims);

    if (n_elem > TENSOR_SCRATCH_LIMIT) {
        fprintf(stderr,
                "create_tensor: want %zu elements (limit %u)\n",
                n_elem, (unsigned)TENSOR_SCRATCH_LIMIT);
        abort();
    }

    Tensor t = {0};
    t.dims   = (uint8_t)dims;
    t.size   = n_elem;

    t.shape = malloc(dims * sizeof(uint8_t));
    if (!t.shape) { perror("create_tensor(shape)"); exit(EXIT_FAILURE); }
    memcpy(t.shape, shape, dims * sizeof(uint8_t));

    t.q_data = malloc(n_elem * sizeof(int8_t));
    if (!t.q_data) { perror("create_tensor(q_data)"); exit(EXIT_FAILURE); }

    t.f_data = NULL;

#ifdef DEBUG
    fprintf(stderr, "[create_tensor] size=%zu  q_data=%p  f_data=%p\n",
            t.size, (void*)t.q_data, (void*)t.f_data);
#endif
    return t;
}

/* Clone tensor: duplicates shape + existing q_data/f_data only */
Tensor clone_tensor(const Tensor *src)
{
    Tensor dst = create_tensor(src->shape, src->dims);

    if (src->f_data) {
        ensure_fp32(&dst);
        memcpy(dst.f_data, src->f_data, dst.size * sizeof(float));
    }
    if (src->q_data) {
        dst.q_data = malloc(dst.size * sizeof(int8_t));
        if (!dst.q_data) { perror("clone_tensor(q)"); exit(EXIT_FAILURE); }
        memcpy(dst.q_data, src->q_data, dst.size * sizeof(int8_t));
    }
    return dst;
}

/* Free all tensor memory and zero the struct */
void free_tensor(Tensor *t)
{
    if (!t) return;
    free(t->shape);
    free(t->f_data);
    free(t->q_data);
    memset(t, 0, sizeof(*t));
}

/* Load tensor from binary file (header + payload), format auto-detected */
static Tensor _load_tensor_generic(const char *path,
                                   int dims,
                                   int expect_float_data)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(EXIT_FAILURE); }

    float hdr_f[8] = {0};
    if (fread(hdr_f, sizeof(float), dims, f) != (size_t)dims) {
        fprintf(stderr, "tensor: header too short in %s\n", path);
        exit(EXIT_FAILURE);
    }

    uint8_t shape8[8];
    int uint8_header = 1;
    for (int i = 0; i < dims; ++i) {
        if (hdr_f[i] != floorf(hdr_f[i]) || hdr_f[i] > 255.f)
            uint8_header = 0;
        shape8[i] = (uint8_t)hdr_f[i];
    }
    if (!uint8_header)
        for (int i = 0; i < dims; ++i)
            shape8[i] = (uint8_t)hdr_f[i];

    Tensor t = create_tensor(shape8, dims);
    size_t n = t.size;

    if (expect_float_data) {
        if (fread(t.f_data, sizeof(float), n, f) != n) {
            fprintf(stderr, "tensor: data truncated in %s\n", path);
            exit(EXIT_FAILURE);
        }
    } else {
        if (fread(t.q_data, sizeof(int8_t), n, f) != n) {
            fprintf(stderr, "tensor: data truncated in %s\n", path);
            exit(EXIT_FAILURE);
        }
    }
    fclose(f);
    return t;
}

/* Public wrappers */
Tensor f_load_tensor(const char *p, int d) { return _load_tensor_generic(p, d, 1); }
Tensor  load_tensor(const char *p, int d) { return _load_tensor_generic(p, d, 0); }

/* Save tensor to file: shape + data (either float or int8) */
void save_tensor(const char *path, const Tensor *t)
{
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(EXIT_FAILURE); }

    fwrite(t->shape, sizeof(uint8_t), t->dims, f);

    if (t->f_data)
        fwrite(t->f_data, sizeof(float),  t->size, f);
    else
        fwrite(t->q_data, sizeof(int8_t), t->size, f);

    fclose(f);
}

/* Print tensor shape */
void print_shape(const Tensor *t)
{
    printf("Shape[");
    for (int i = 0; i < t->dims; ++i) {
        printf("%d", t->shape[i]);
        if (i != t->dims - 1) putchar(',');
    }
    printf("] (dims=%d)\n", t->dims);
}

/* Print first N elements of the tensor for inspection */
void print_tensor(const Tensor *t)
{
    if (!t || t->size == 0) {
        printf("[Empty tensor]\n");
        return;
    }

    printf("Tensor size = %zu  ", t->size);
    print_shape(t);

    int N = (t->size < 8) ? (int)t->size : 8;
    if (t->f_data) {
        printf("float data: ");
        for (int i = 0; i < N; ++i)
            printf("%.3f%s", t->f_data[i], (i != N - 1) ? ", " : "");
    } else if (t->q_data) {
        printf("int8 data: ");
        for (int i = 0; i < N; ++i)
            printf("%d%s", t->q_data[i], (i != N - 1) ? ", " : "");
    } else {
        printf("No data allocated");
    }
    printf("\n");
}
