#include "func_cuda.h"

void check(cudaError_t err, const char *funcName, const char *file, const int line)
{
        if (err != cudaSuccess) {
                fprintf(stderr, "Error: %s:%d \"%s\": %s\n",
                        file, line,
                        funcName, cudaGetErrorString(err));
                cudaDeviceReset();
                exit(EXIT_FAILURE);
        }
}

int get_device_name_by_id(int id, char *name, int len)
{
        int deviceCount;

        checkCudaError(cudaGetDeviceCount(&deviceCount));

        if (id >= deviceCount || id < 0) {
                return -1;
        }
        else {
                struct cudaDeviceProp prop;

                checkCudaError(cudaGetDeviceProperties(&prop, id));
                if (len - 1 > strlen(prop.name))
                        strcpy(name, prop.name);
                else
                        return 1;
        }

        return 0;
}

int get_device_id_by_name(const char *name, int *list, size_t size)
{
        int find = 0;

        int deviceCount;
        checkCudaError(cudaGetDeviceCount(&deviceCount));

        struct cudaDeviceProp prop;
        for (int i = 0; i < deviceCount; i++) {
                checkCudaError(cudaGetDeviceProperties(&prop, i));

                if (strstr(prop.name, name) != NULL)
                        if (--size >= 0)
                                list[find++] = i;
                        else
                                return -1;
                else
                        continue;
        }

        return find;
}

int getNameById(int id, char *name, size_t len)
{
        int deviceCount;

        checkCudaError(cudaGetDeviceCount(&deviceCount));

        if (id >= deviceCount || id < 0) {
                return -1;
        }
        else {
                struct cudaDeviceProp prop;

                checkCudaError(cudaGetDeviceProperties(&prop, id));
                if (len - 1 > strlen(prop.name))
                        strcpy(name, prop.name);
                else
                        return 1;
        }

        return 0;
}

int getIdByName(const char *name, int *list, size_t size)
{
        int find = 0;

        int deviceCount;
        checkCudaError(cudaGetDeviceCount(&deviceCount));

        struct cudaDeviceProp prop;
        for (int i = 0; i < deviceCount; i++) {
                checkCudaError(cudaGetDeviceProperties(&prop, i));

                if (strstr(prop.name, name) != NULL)
                        if (--size >= 0)
                                list[find++] = i;
                        else
                                return -1;
                else
                        continue;
        }

        return find;
}
