#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "func_cuda.h"
#include "parser.h"

extern void hostToDevice(int numGB, int *devices, int numGPUs);
extern void deviceToHost(int numGB, int *devices, int numGPUs);
extern void bidHostDevice(int numGB, int *devices, int numGPUs);
extern void unidp2p(int size, int src, int dst);
extern void bidp2p(int size, int src, int dst);

static void printHelp(char *progname)
{
    printf("Usage:\n");
    printf("\t%s <h2d|d2h|bid|p2p|bidp2p>\n", progname);
    printf("\t%s [--size <GB>] [--id 0,1,2-4] [--name P100,V100] <h2d|d2h|bid>\n", progname);
    printf("\t%s [--size <GB>] [--src <gpu> --dst <peergpu>] <p2p|bidp2p>\n", progname);
}

int main(int argc, char **argv)
{
    Parser parser;

    parse(argc, argv, &parser);

    if (argc == 1 || parser.help) {
        printHelp(argv[0]);
        return 0;
    }

    /* find GPU */
    int allGPU;
    checkCudaError(cudaGetDeviceCount(&allGPU));
    if (!allGPU) {
        fprintf(stderr, "Error: No CUDA device found\n");
        return 1;
    }

    int testGPUList[allGPU];
    int listSize = 0;

    /* get a list of testing gpu by name or id */
    if (parser.id != NULL) {
        listSize = get_int_range(parser.id, testGPUList, allGPU);
    }
    else if (parser.name != NULL) {
        char *name[allGPU];
        int find, count;

        count = get_str_range(parser.name, name, allGPU);
        for (int i = 0; i < count; i++)
            if (!(find = get_device_id_by_name(name[i], testGPUList + listSize, allGPU - listSize)))
                fprintf(stderr, "Warning: No GPU %s found\n", name[i]);
            else
                listSize += find;
    }
    else {
        listSize = allGPU;
        for (int id = 0; id < allGPU; id++)
            testGPUList[id] = id;
    }

    /* run test */
    switch(parser.method) {
    case h2d:
        hostToDevice(parser.size, testGPUList, listSize);
        break;
    case d2h:
        deviceToHost(parser.size, testGPUList, listSize);
        break;
    case hbd:
        bidHostDevice(parser.size, testGPUList, listSize);
        break;
    case p2p:
        unidp2p(parser.size, parser.src, parser.dst);
        break;
    case pbp:
        bidp2p(parser.size, parser.src, parser.dst);
        break;
    case unknown:
        fprintf(stderr, "Error: Unknown test type\n");
        break;
    }

    // Reset testing GPU
    for (int i = 0; i < listSize; i++)
        cudaSetDevice(testGPUList[i]);

    return 0;
}
