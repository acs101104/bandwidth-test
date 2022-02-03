#ifndef FUNC_CUDA_H
#define FUNC_CUDA_H

#define checkCudaError(val) check((val), #val, __FILE__, __LINE__)

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

void check(cudaError_t err, const char *func_Name, const char *file, const int line);
int get_device_name_by_id(int id, char *GPU_Name, int len);
int get_device_id_by_name(const char *GPU_Name, int *id, size_t size);

#ifdef __cplusplus
}
#endif

#endif
