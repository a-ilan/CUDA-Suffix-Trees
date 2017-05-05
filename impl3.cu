#include "implementation.h"

void impl3(const char* strings, int* indices, int totalLength, int numStrings, int bsize, int bcount){
	Timer timer;
	char* d_strings = NULL;
	int* d_indices = NULL;

	cudaMalloc((void**)&d_strings, sizeof(char)*totalLength);
	cudaMalloc((void**)&indices, sizeof(int)*numStrings);

	cudaMemcpy(d_strings, strings, sizeof(char)*totalLength, cudaMemcpyHostToDevice);
	cudaMemcpy(d_indices, indices, sizeof(int)*numStrings, cudaMemcpyHostToDevice);



	timer.set();

	// put code here

	cout << "running time: " << timer.get() << " ms" << endl;
	
	// free
	cudaFree(d_strings);
	cudaFree(d_indices);
}
