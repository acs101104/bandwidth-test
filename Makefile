CFLAGS = -Wall -g -I $(CUDA_PATH)/include

LDFLAGS = -L $(CUDA_PATH)/lib64 -lcudart

CUDA_PATH ?= /usr/local/cuda

NVCC ?= $(CUDA_PATH)/bin/nvcc

SMS ?= 30 35 37 50 52 60 61 70 75

ifeq ($(GENCODE_FLAGS),)
    $(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
endif

objs = func_cuda.o parser.o gpu.o htod.o p2p.o delay.o

all: cudaBandwidth

cudaBandwidth: main.c $(objs)
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

delay.o: delay.cu
	$(NVCC) $(GENCODE_FLAGS) -c -o $@ $<

htod.o: htod.cu func_cuda.h gpu.h
	$(NVCC) $(GENCODE_FLAGS) -c -o $@ $<

p2p.o: p2p.cu func_cuda.h gpu.h
	$(NVCC) $(GENCODE_FLAGS) -c -o $@ $<

# header dependency
gpu.o: gpu.h func_cuda.h
parser.o: parser.h
func_cuda.o: func_cuda.h

clean:
	$(RM) cudaBandwidth $(objs)

.PHONY: all clean
