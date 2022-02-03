#ifndef GPU_H
#define GPU_H

#include <stdlib.h>
#include <cuda_runtime.h>

#define ALLOC_SIZE      1000000000UL
#define MAX_GPU_NAME    64

typedef enum { htod, dtoh, bid_hd, p2p_src, p2p_dst, bid_p2p } GPU_TYPE;

typedef struct {
    int id;
    char name[MAX_GPU_NAME];
    unsigned char *h_idata;
    unsigned char *h_odata;
    unsigned char *d_idata;
    unsigned char *d_odata;

    cudaStream_t stream;
    cudaEvent_t start;
    cudaEvent_t stop;

    cudaStream_t stream1;
    cudaEvent_t start1;
    cudaEvent_t stop1;
} GPU_t;


#ifdef __cplusplus
extern "C" {
#endif

void initGPUResource(GPU_t *, GPU_TYPE, int id);
void freeGPUResource(GPU_t *, GPU_TYPE);
void verifyData(GPU_t *);

#ifdef __cplusplus
}
#endif

#endif
