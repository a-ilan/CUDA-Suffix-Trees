#include "implementation.h"

void impl3(char* text, int* indices, int totalLength, int numStrings, int bsize, int bcount){
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
