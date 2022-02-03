#include <stdio.h>
#include "gpu.h"
#include "func_cuda.h"

#define ALLOC_SIZE      1000000000UL

extern __global__ void
delay(volatile int *flag, unsigned long long timeout_clocks = 1000000000);

static int checkp2p(int gpu, int peergpu) {
    int access;
    checkCudaError(cudaDeviceCanAccessPeer(&access, gpu, peergpu));
    return access;
}

void unidp2p(int size, int src, int dst)
{
    printf("Testing memory copy from GPU %d to GPU %d\n", src, dst);

    int p2p_enable;
    GPU_t gpu, peergpu;

    initGPUResource(&gpu, p2p_src, src);
    initGPUResource(&peergpu, p2p_dst, dst);

    if (p2p_enable = checkp2p(gpu.id, peergpu.id)) {
        checkCudaError(cudaSetDevice(gpu.id));
        checkCudaError(cudaDeviceEnablePeerAccess(peergpu.id, 0));
        checkCudaError(cudaSetDevice(peergpu.id));
        checkCudaError(cudaDeviceEnablePeerAccess(gpu.id, 0));
    }
    else {
        fprintf(stderr, "Warning: GPU %d can't directly access GPU %d\n", src, dst);
    }

    volatile int *flag = NULL;
    checkCudaError(cudaHostAlloc((void **)&flag, sizeof(*flag), cudaHostAllocPortable));

    // Arrange task to stream
    printf("Arrange GPU tasks\n");

    *flag = 0;
    delay<<<1, 1, 0, gpu.stream>>>(flag);
    checkCudaError(cudaEventRecord(gpu.start, gpu.stream));
    for (int i = 0; i < size; i++)
        checkCudaError(cudaMemcpyAsync(peergpu.d_odata, gpu.d_idata, ALLOC_SIZE,
                       cudaMemcpyDefault, gpu.stream));
    checkCudaError(cudaEventRecord(gpu.stop, gpu.stream));

    // Release stream
    *flag = 1;
    checkCudaError(cudaStreamSynchronize(gpu.stream));

    // Output result and free resources
    float time_ms, time, bandwidth;

    checkCudaError(cudaEventElapsedTime(&time_ms, gpu.start, gpu.stop));
    time = time_ms / (float) 1e3;
    bandwidth = size / time;
    printf("%20s,%15s\n", "Bandwidth (GB/s)", "Latency (s)");
    printf("%20.2f,%15.2f\n", bandwidth, time / size);

    if (p2p_enable) {
        checkCudaError(cudaSetDevice(gpu.id));
        checkCudaError(cudaDeviceDisablePeerAccess(peergpu.id));

        checkCudaError(cudaSetDevice(peergpu.id));
        checkCudaError(cudaDeviceDisablePeerAccess(gpu.id));
    }

    freeGPUResource(&gpu, p2p_src);
    freeGPUResource(&peergpu, p2p_dst);

    checkCudaError(cudaFreeHost((void *) flag));
}

void bidp2p(int size, int src, int dst)
{
    printf("Testing memory copy between GPU %d and GPU %d\n", src, dst);

    int p2p_enable;
    GPU_t gpu, peergpu;

    initGPUResource(&gpu, bid_p2p, src);
    initGPUResource(&peergpu, bid_p2p, dst);

    if (p2p_enable = checkp2p(gpu.id, peergpu.id)) {
        checkCudaError(cudaSetDevice(gpu.id));
        checkCudaError(cudaDeviceEnablePeerAccess(peergpu.id, 0));
        checkCudaError(cudaSetDevice(peergpu.id));
        checkCudaError(cudaDeviceEnablePeerAccess(gpu.id, 0));
    }
    else {
        fprintf(stderr, "Warning: GPU: %d can't directly access GPU: %d\n", src, dst);
    }

    volatile int *flag = NULL;
    checkCudaError(cudaHostAlloc((void **)&flag, sizeof(*flag), cudaHostAllocPortable));

    // Arrange task to each stream
    printf("Arrange GPU tasks\n");

    *flag = 0;
    delay<<<1, 1, 0, gpu.stream>>>(flag);
    checkCudaError(cudaEventRecord(gpu.start, gpu.stream));
    checkCudaError(cudaStreamWaitEvent(peergpu.stream, gpu.start, 0));
    for (int i = 0; i < size; i++) {
        checkCudaError(cudaMemcpyAsync(peergpu.d_odata, gpu.d_idata, ALLOC_SIZE, cudaMemcpyDefault, gpu.stream));
        checkCudaError(cudaMemcpyAsync(gpu.d_odata, peergpu.d_idata, ALLOC_SIZE, cudaMemcpyDefault, peergpu.stream));
    }
    checkCudaError(cudaEventRecord(peergpu.stop, peergpu.stream));
    checkCudaError(cudaStreamWaitEvent(gpu.stream, peergpu.stop, 0));
    checkCudaError(cudaEventRecord(gpu.stop, gpu.stream));

    *flag = 1;
    checkCudaError(cudaStreamSynchronize(gpu.stream));
    checkCudaError(cudaStreamSynchronize(peergpu.stream));

    // Output result and free resources
    float time_ms, time, bandwidth;

    checkCudaError(cudaEventElapsedTime(&time_ms, gpu.start, gpu.stop));
    time = time_ms / (float) 1e3;
    bandwidth = 2 * size / time;
    printf("%20s,%15s\n", "Bandwidth (GB/s)", "Latency (s)");
    printf("%20.2f,%15.2f\n", bandwidth, time / size);

    if (p2p_enable) {
        checkCudaError(cudaSetDevice(gpu.id));
        checkCudaError(cudaDeviceDisablePeerAccess(peergpu.id));
        checkCudaError(cudaSetDevice(peergpu.id));
        checkCudaError(cudaDeviceDisablePeerAccess(gpu.id));
    }

    freeGPUResource(&peergpu, p2p_dst);
    freeGPUResource(&gpu, p2p_src);

    checkCudaError(cudaFreeHost((void *) flag));
}
