
.PHONY: build clean clobber run

all: build

build: conway

conway: conway.cu
	nvcc -I. -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=compute_30 -gencode arch=compute_35,code=sm_35 -lineinfo -O3 -o conway conway.cu

clean:
	-rm conway

clobber: clean


