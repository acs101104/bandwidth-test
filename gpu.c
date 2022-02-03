#include "gpu.h"
#include "func_cuda.h"

void initGPUResource(GPU_t *gpu, GPU_TYPE type, int id)
{
        gpu->id = id;

        struct cudaDeviceProp prop;
        checkCudaError(cudaGetDeviceProperties(&prop, id));
        strcpy(gpu->name, prop.name);

        switch(type) {
        case htod:
                checkCudaError(cudaSetDevice(gpu->id));
                checkCudaError(cudaMallocHost((void **) &gpu->h_idata, ALLOC_SIZE));
                checkCudaError(cudaMalloc((void **) &gpu->d_odata, ALLOC_SIZE));
                checkCudaError(cudaStreamCreateWithFlags(&gpu->stream, cudaStreamNonBlocking));
                checkCudaError(cudaEventCreate(&gpu->start));
                checkCudaError(cudaEventCreate(&gpu->stop));
                for (int i = 0; i < ALLOC_SIZE; ++i)
                        (gpu->h_idata)[i] = (unsigned char)(i & 0xff);
                break;
        case dtoh:
                checkCudaError(cudaSetDevice(gpu->id));
                checkCudaError(cudaMallocHost((void **) &gpu->h_idata, ALLOC_SIZE));
                checkCudaError(cudaMallocHost((void **) &gpu->h_odata, ALLOC_SIZE));
                checkCudaError(cudaMalloc((void **) &gpu->d_idata, ALLOC_SIZE));
                checkCudaError(cudaStreamCreateWithFlags(&gpu->stream, cudaStreamNonBlocking));
                checkCudaError(cudaEventCreate(&gpu->start));
                checkCudaError(cudaEventCreate(&gpu->stop));
                for (int i = 0; i < ALLOC_SIZE; ++i)
                        (gpu->h_idata)[i] = (unsigned char)(i & 0xff);
                checkCudaError(cudaMemcpy(gpu->d_idata, gpu->h_idata, ALLOC_SIZE, cudaMemcpyDefault));
                break;
        case bid_hd:
                checkCudaError(cudaSetDevice(gpu->id));
                checkCudaError(cudaMallocHost((void **) &gpu->h_idata, ALLOC_SIZE));
                checkCudaError(cudaMallocHost((void **) &gpu->h_odata, ALLOC_SIZE));
                checkCudaError(cudaMalloc((void **) &gpu->d_idata, ALLOC_SIZE));
                checkCudaError(cudaMalloc((void **) &gpu->d_odata, ALLOC_SIZE));
                checkCudaError(cudaStreamCreateWithFlags(&gpu->stream, cudaStreamNonBlocking));
                checkCudaError(cudaStreamCreateWithFlags(&gpu->stream1, cudaStreamNonBlocking));
                checkCudaError(cudaEventCreate(&gpu->start));
                checkCudaError(cudaEventCreate(&gpu->start1));
                checkCudaError(cudaEventCreate(&gpu->stop));
                checkCudaError(cudaEventCreate(&gpu->stop1));
                for (int i = 0; i < ALLOC_SIZE; ++i)
                        (gpu->h_idata)[i] = (unsigned char)(i & 0xff);
                checkCudaError(cudaMemcpy(gpu->d_idata, gpu->h_idata, ALLOC_SIZE, cudaMemcpyDefault));
                break;
        case p2p_src:
                checkCudaError(cudaSetDevice(gpu->id));
                checkCudaError(cudaMallocHost((void **) &gpu->h_idata, ALLOC_SIZE));
                checkCudaError(cudaMalloc((void **) &gpu->d_idata, ALLOC_SIZE));
                checkCudaError(cudaStreamCreateWithFlags(&gpu->stream, cudaStreamNonBlocking));
                checkCudaError(cudaEventCreate(&gpu->start));
                checkCudaError(cudaEventCreate(&gpu->stop));
                for (int i = 0; i < ALLOC_SIZE; ++i)
                        (gpu->h_idata)[i] = (unsigned char)(i & 0xff);
                checkCudaError(cudaMemcpy(gpu->d_idata, gpu->h_idata, ALLOC_SIZE, cudaMemcpyDefault));
                break;
        case p2p_dst:
                checkCudaError(cudaSetDevice(gpu->id));
                checkCudaError(cudaMalloc((void **) &gpu->d_odata, ALLOC_SIZE));
                break;
        case bid_p2p:
                checkCudaError(cudaSetDevice(gpu->id));
                checkCudaError(cudaMallocHost((void **) &gpu->h_idata, ALLOC_SIZE));
                checkCudaError(cudaMalloc((void **) &gpu->d_idata, ALLOC_SIZE));
                checkCudaError(cudaMalloc((void **) &gpu->d_odata, ALLOC_SIZE));
                checkCudaError(cudaStreamCreateWithFlags(&gpu->stream, cudaStreamNonBlocking));
                checkCudaError(cudaEventCreate(&gpu->start));
                checkCudaError(cudaEventCreate(&gpu->stop));
                for (int i = 0; i < ALLOC_SIZE; ++i)
                        (gpu->h_idata)[i] = (unsigned char)(i & 0xff);
                checkCudaError(cudaMemcpy(gpu->d_idata, gpu->h_idata, ALLOC_SIZE, cudaMemcpyDefault));
                break;
        default:
                // TODO
                break;
        }
}

void freeGPUResource(GPU_t *gpu, GPU_TYPE type)
{
        switch(type) {
        case htod:
                checkCudaError(cudaSetDevice(gpu->id));
                checkCudaError(cudaFreeHost(gpu->h_idata));
                checkCudaError(cudaFree(gpu->d_odata));
                checkCudaError(cudaStreamDestroy(gpu->stream));
                checkCudaError(cudaEventDestroy(gpu->start));
                checkCudaError(cudaEventDestroy(gpu->stop));
                break;
        case dtoh:
                checkCudaError(cudaSetDevice(gpu->id));
                checkCudaError(cudaFreeHost(gpu->h_idata));
                checkCudaError(cudaFreeHost(gpu->h_odata));
                checkCudaError(cudaFree(gpu->d_idata));
                checkCudaError(cudaStreamDestroy(gpu->stream));
                checkCudaError(cudaEventDestroy(gpu->start));
                checkCudaError(cudaEventDestroy(gpu->stop));
                break;
        case bid_hd:
                checkCudaError(cudaSetDevice(gpu->id));
                checkCudaError(cudaFreeHost(gpu->h_idata));
                checkCudaError(cudaFreeHost(gpu->h_odata));
                checkCudaError(cudaFree(gpu->d_idata));
                checkCudaError(cudaFree(gpu->d_odata));
                checkCudaError(cudaStreamDestroy(gpu->stream));
                checkCudaError(cudaStreamDestroy(gpu->stream1));
                checkCudaError(cudaEventDestroy(gpu->start));
                checkCudaError(cudaEventDestroy(gpu->start1));
                checkCudaError(cudaEventDestroy(gpu->stop));
                checkCudaError(cudaEventDestroy(gpu->stop1));
                break;
        case p2p_src:
                checkCudaError(cudaSetDevice(gpu->id));
                checkCudaError(cudaFreeHost(gpu->h_idata));
                checkCudaError(cudaFree(gpu->d_idata));
                checkCudaError(cudaStreamDestroy(gpu->stream));
                checkCudaError(cudaEventDestroy(gpu->start));
                checkCudaError(cudaEventDestroy(gpu->stop));
                break;
        case p2p_dst:
                checkCudaError(cudaSetDevice(gpu->id));
                checkCudaError(cudaFree(gpu->d_odata));
                break;
        case bid_p2p:
                checkCudaError(cudaSetDevice(gpu->id));
                checkCudaError(cudaFreeHost(gpu->h_idata));
                checkCudaError(cudaFree(gpu->d_idata));
                checkCudaError(cudaFree(gpu->d_odata));
                checkCudaError(cudaStreamDestroy(gpu->stream));
                checkCudaError(cudaEventDestroy(gpu->start));
                checkCudaError(cudaEventDestroy(gpu->stop));
                break;
        default:
                // TODO
                break;
        }
}

inline void verifyData(GPU_t *gpu)
{
        for (int i = 0; i < 100; i++) {
                printf("%d,", (gpu->h_idata)[i]);
                if ((i + 1) % 10 == 0)
                        printf("\n");
        }
}
