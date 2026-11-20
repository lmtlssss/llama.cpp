// A row of src1 must give the same bits whether it is multiplied alone (one token: generation, or the
// last prompt token after a prompt-cache hit) or inside a batch (prompt processing).
// Shapes are those of SmolLM2-135M (n_embd 576, n_embd_k_gqa 192, n_ff 1536).

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int n_failed = 0;

static uint32_t rng_state = 1234;

static float rnd(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return (float)(rng_state >> 8) / (float)(1u << 24) * 2.0f - 1.0f;
}

static void cpu_set_n_threads(ggml_backend_t backend, int n_threads) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    void (* set_n_threads)(ggml_backend_t, int) =
        (void (*)(ggml_backend_t, int)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (!set_n_threads) {
        fprintf(stderr, "ggml_backend_set_n_threads not available\n");
        exit(1);
    }
    set_n_threads(backend, n_threads);
}

// w is n_in x n_out of type wtype, x is n_in x n_batch F32
static void test_rows(ggml_backend_t backend, ggml_type wtype, int64_t n_in, int64_t n_out, int64_t n_batch, int n_threads) {
    rng_state = 1234;
    float * w = (float *) malloc(n_in * n_out * sizeof(float));
    float * x = (float *) malloc(n_in * n_batch * sizeof(float));
    for (int64_t i = 0; i < n_in * n_out; ++i) {
        w[i] = rnd();
    }
    for (int64_t i = 0; i < n_in * n_batch; ++i) {
        x[i] = rnd();
    }

    struct ggml_init_params ip = { 16*ggml_tensor_overhead() + 2*ggml_graph_overhead(), NULL, true };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * tw  = ggml_new_tensor_2d(ctx, wtype,         n_in, n_out);
    struct ggml_tensor * txb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_in, n_batch);
    struct ggml_tensor * tx1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_in, 1);
    struct ggml_tensor * yb  = ggml_mul_mat(ctx, tw, txb);
    struct ggml_tensor * y1  = ggml_mul_mat(ctx, tw, tx1);

    struct ggml_cgraph * gb = ggml_new_graph(ctx);
    ggml_build_forward_expand(gb, yb);
    struct ggml_cgraph * g1 = ggml_new_graph(ctx);
    ggml_build_forward_expand(g1, y1);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    void * wq = malloc(ggml_nbytes(tw));
    ggml_quantize_chunk(wtype, w, wq, 0, n_out, n_in, NULL);
    ggml_backend_tensor_set(tw, wq, 0, ggml_nbytes(tw));
    ggml_backend_tensor_set(txb, x, 0, ggml_nbytes(txb));

    cpu_set_n_threads(backend, n_threads);
    ggml_backend_graph_compute(backend, gb);
    float * outb = (float *) malloc(ggml_nbytes(yb));
    ggml_backend_tensor_get(yb, outb, 0, ggml_nbytes(yb));

    const int64_t rows[3] = { 0, n_batch / 2, n_batch - 1 };
    float * out1 = (float *) malloc(ggml_nbytes(y1));
    int64_t  n_diff  = 0;
    int64_t  first_r = -1;
    int64_t  first_i = -1;
    uint32_t first_b = 0;
    uint32_t first_a = 0;
    for (int ir = 0; ir < 3; ++ir) {
        const int64_t r = rows[ir];
        ggml_backend_tensor_set(tx1, x + r * n_in, 0, ggml_nbytes(tx1));
        ggml_backend_graph_compute(backend, g1);
        ggml_backend_tensor_get(y1, out1, 0, ggml_nbytes(y1));
        for (int64_t i = 0; i < n_out; ++i) {
            if (memcmp(&out1[i], &outb[r * n_out + i], sizeof(float)) != 0) {
                if (first_r < 0) {
                    first_r = r;
                    first_i = i;
                    memcpy(&first_b, &outb[r * n_out + i], sizeof(float));
                    memcpy(&first_a, &out1[i],             sizeof(float));
                }
                ++n_diff;
            }
        }
    }

    printf("TEST %s %lldx%lld, batch %lld, threads %d: %s", ggml_type_name(wtype),
           (long long) n_in, (long long) n_out, (long long) n_batch, n_threads, n_diff == 0 ? "PASS" : "FAIL");
    if (n_diff != 0) {
        printf(" (%lld of %lld values differ, first at row %lld element %lld: batch %08x, alone %08x)",
               (long long) n_diff, (long long) (3 * n_out), (long long) first_r, (long long) first_i, first_b, first_a);
        ++n_failed;
    }
    printf("\n");

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    free(w); free(x); free(wq); free(outb); free(out1);
}

int main(void) {
    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    if (!backend) {
        fprintf(stderr, "failed to initialize the CPU backend\n");
        return 1;
    }

    const int64_t shapes[4][2] = { { 576, 576 }, { 576, 192 }, { 576, 1536 }, { 1536, 576 } };
    const int64_t batches[3]   = { 2, 16, 512 };
    const int     threads[2]   = { 1, 4 };
    for (int it = 0; it < 2; ++it) {
        for (int is = 0; is < 4; ++is) {
            for (int ib = 0; ib < 3; ++ib) {
                test_rows(backend, GGML_TYPE_F16, shapes[is][0], shapes[is][1], batches[ib], threads[it]);
            }
        }
    }

    ggml_backend_free(backend);
    printf("%s\n", n_failed == 0 ? "OK" : "FAILED");
    return n_failed == 0 ? 0 : 1;
}
