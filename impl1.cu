#include "implementation.h"

__global__ void suffix_tree_construction(const char* text, int* indices, int totalLength, int numStrings){
	const int tid = threadIdx.x + blockDim.x*blockIdx.x;
	const int nThreads = blockDim.x*gridDim.x;
	const int iter = totalLength%nThreads == 0? totalLength/nThreads : totalLength/nThreads+1;

	for(int i = 0; i < iter; i++){
		int dataid = tid + i*nThreads;
		if(dataid < totalLength){
			
		}
	}
}

void impl1(const char* text, int* indices, int totalLength, int numStrings, int bsize, int bcount){
	Timer timer;
        char* d_text = NULL;
        int* d_indices = NULL;

        cudaMalloc((void**)&d_text, sizeof(char)*totalLength);
        cudaMalloc((void**)&d_indices, sizeof(int)*numStrings);

        cudaMemcpy(d_text, text, sizeof(char)*totalLength, cudaMemcpyHostToDevice);
        cudaMemcpy(d_indices, indices, sizeof(int)*numStrings, cudaMemcpyHostToDevice);

	timer.set();

	// put code here

	cout << "running time: " << timer.get() << " ms" << endl;

	// free
	cudaFree(d_text);
	cudaFree(d_indices);
}
