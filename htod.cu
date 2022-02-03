#include <stdio.h>
#include "gpu.h"
#include "func_cuda.h"

extern __global__ void
delay(volatile int *flag, unsigned long long timeout_clocks = 1000000000);

void hostToDevice(int numGB, int *devices, int numGPUs)
{
    printf("Memcpy copy from host to GPU\n");

    GPU_t gpu[numGPUs];

    for (int i = 0; i < numGPUs; i++)
        initGPUResource(&gpu[i], htod, devices[i]);

    volatile int *flag = NULL;
    checkCudaError(cudaHostAlloc((void **)&flag, sizeof(*flag), cudaHostAllocPortable));

    // Arrange tasks to stream queue
    *flag = 0;
    for (int i = 0; i < numGPUs; i++) {
        printf("Arrange GPU %d tasks\n", gpu[i].id);

        checkCudaError(cudaSetDevice(gpu[i].id));
        delay<<<1, 1, 0, gpu[i].stream>>>(flag);
        checkCudaError(cudaEventRecord(gpu[i].start, gpu[i].stream));
        for (int r = 0; r < numGB; r++)
            checkCudaError(cudaMemcpyAsync(gpu[i].d_odata, gpu[i].h_idata, ALLOC_SIZE,
                                           cudaMemcpyDefault, gpu[i].stream));
        checkCudaError(cudaEventRecord(gpu[i].stop, gpu[i].stream));
    }

    // Release queued tasks
    printf("Starting transfer %d GB...\n", numGB);
    *flag = 1;
    for (int i = 0; i < numGPUs; ++i) {
        checkCudaError(cudaSetDevice(gpu[i].id));
        checkCudaError(cudaStreamSynchronize(gpu[i].stream));
    }

    // Output result and free resources
    checkCudaError(cudaFreeHost((void *) flag));
    float elapsedTimeInMs, bandwidthInGBs;

    printf("%10s,%25s,%20s,%15s\n", "GPU ID", "Name", "Bandwidth (GB/s)", "Latency (s)");
    for (int i = 0; i < numGPUs; ++i) {
        checkCudaError(cudaEventElapsedTime(&elapsedTimeInMs, gpu[i].start, gpu[i].stop));
        bandwidthInGBs = numGB / (elapsedTimeInMs / (float) 1e3);
        printf("%10d,%25s,%20.2f,%15.2f\n",
                gpu[i].id,
                gpu[i].name,
                bandwidthInGBs,
                elapsedTimeInMs / numGB / (float) 1e3);
        freeGPUResource(&gpu[i], htod);
    }
}

void deviceToHost(int numGB, int *devices, int numGPUs)
{
    printf("Memory copy from GPU to host\n");

    GPU_t gpu[numGPUs];

    for (int i = 0; i < numGPUs; ++i)
        initGPUResource(&gpu[i], dtoh, devices[i]);

    volatile int *flag = NULL;
    checkCudaError(cudaHostAlloc((void **)&flag, sizeof(*flag), cudaHostAllocPortable));

    // Arrange tasks to stream queue
    *flag = 0;
    for (int i = 0; i < numGPUs; i++) {
        printf("Arrange GPU %d tasks\n", gpu[i].id);

        checkCudaError(cudaSetDevice(gpu[i].id));
        delay<<<1, 1, 0, gpu[i].stream>>>(flag);
        checkCudaError(cudaEventRecord(gpu[i].start, gpu[i].stream));
        for (int r = 0; r < numGB; r++)
            checkCudaError(cudaMemcpyAsync(gpu[i].h_odata, gpu[i].d_idata, ALLOC_SIZE,
                                           cudaMemcpyDefault, gpu[i].stream));
        checkCudaError(cudaEventRecord(gpu[i].stop, gpu[i].stream));
    }

    // Release queued tasks
    printf("Starting transfer %d GB...\n", numGB);
    *flag = 1;
    for (int i = 0; i < numGPUs; i++) {
        checkCudaError(cudaSetDevice(gpu[i].id));
        checkCudaError(cudaStreamSynchronize(gpu[i].stream));
    }

    // Output result and free resources
    checkCudaError(cudaFreeHost((void *) flag));
    float elapsedTimeInMs, bandwidthInGBs;

    printf("%10s,%25s,%20s,%15s\n", "GPU ID", "Name", "Bandwidth (GB/s)", "Latency (s)");
    for (int i = 0; i < numGPUs; i++) {
        checkCudaError(cudaEventElapsedTime(&elapsedTimeInMs, gpu[i].start, gpu[i].stop));
        bandwidthInGBs = numGB / (elapsedTimeInMs / (float) 1e3);
        printf("%10d,%25s,%20.2f,%15.2f\n",
                gpu[i].id,
                gpu[i].name,
                bandwidthInGBs,
                elapsedTimeInMs / numGB / (float) 1e3);
        freeGPUResource(&gpu[i], dtoh);
    }
}

void bidHostDevice(int numGB, int *devices, int numGPUs)
{
    printf("Memory copy between host and GPU\n");

    GPU_t gpu[numGPUs];

    for (int i = 0; i < numGPUs; i++)
        initGPUResource(&gpu[i], bid_hd, devices[i]);

    volatile int *flag = NULL;
    checkCudaError(cudaHostAlloc((void **)&flag, sizeof(*flag), cudaHostAllocPortable));

    // Arrange tasks to stream queue
    *flag = 0;
    for (int i = 0; i < numGPUs; i++) {
        printf("Arrange GPU %d tasks\n", gpu[i].id);
        checkCudaError(cudaSetDevice(gpu[i].id));

        delay<<<1, 1, 0, gpu[i].stream>>>(flag);
        checkCudaError(cudaEventRecord(gpu[i].start, gpu[i].stream));
        checkCudaError(cudaStreamWaitEvent(gpu[i].stream1, gpu[i].start, 0));

        for (int r = 0; r < numGB; r++) {
            checkCudaError(cudaMemcpyAsync(gpu[i].d_odata, gpu[i].h_idata, ALLOC_SIZE,
                                           cudaMemcpyDefault, gpu[i].stream));
            checkCudaError(cudaMemcpyAsync(gpu[i].h_odata, gpu[i].d_idata, ALLOC_SIZE,
                                           cudaMemcpyDefault, gpu[i].stream1));
        }
        checkCudaError(cudaEventRecord(gpu[i].stop1, gpu[i].stream1));
        checkCudaError(cudaStreamWaitEvent(gpu[i].stream, gpu[i].stop1, 0));
        checkCudaError(cudaEventRecord(gpu[i].stop, gpu[i].stream));
    }

    // Release queued tasks
    printf("Starting transfer %d GB...\n", numGB);
    *flag = 1;
    for (int i = 0; i < numGPUs; i++) {
        checkCudaError(cudaSetDevice(gpu[i].id));
        checkCudaError(cudaStreamSynchronize(gpu[i].stream));
        checkCudaError(cudaStreamSynchronize(gpu[i].stream1));
    }

    // Output result and free resources
    checkCudaError(cudaFreeHost((void *) flag));
    float elapsedTimeInMs, bandwidthInGBs;

    printf("%10s,%25s,%20s,%15s\n", "GPU ID", "Name", "Bandwidth (GB/s)", "Latency (s)");
    for (int i = 0; i < numGPUs; i++) {
        checkCudaError(cudaEventElapsedTime(&elapsedTimeInMs, gpu[i].start, gpu[i].stop));
        bandwidthInGBs = 2 * numGB / (elapsedTimeInMs / 1e3);
        printf("%10d,%25s,%20.2f,%15.2f\n",
                gpu[i].id,
                gpu[i].name,
                bandwidthInGBs,
                elapsedTimeInMs / numGB / (float) 1e3);
        freeGPUResource(&gpu[i], bid_hd);
    }
}
