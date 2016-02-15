all:
	/usr/local/cuda/bin/nvcc -O3 -Xcompiler -use_fast_math -Xcompiler -fopenmp -gencode arch=compute_30,code=sm_30 -m64 -lgomp -lcurand main.cpp sgd.cu -o sgd

