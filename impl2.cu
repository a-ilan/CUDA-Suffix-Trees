#include "implementation.h"

__device__ int getEndOfSuffix(int start, const char* text){
	int i = start;
	while(text[i] != '$') i++;
	return i+1;
}

__global__ void constructSuffixTree(Node* root, 
		const char* text, int* indices, int* suffixes, 
		int totalLength, int numStrings, int numSuffixes){
	const int tid = threadIdx.x + blockDim.x*blockIdx.x;
	const int nThreads = blockDim.x*gridDim.x;
	const int iter = numSuffixes%nThreads == 0? numSuffixes/nThreads : numSuffixes/nThreads+1;

	for(int i = 0; i < iter; i++){
		int dataid = tid + i*nThreads;
		if(dataid < numSuffixes){
			int start = suffixes[dataid];
			int end = getEndOfSuffix(start,text);

			char character = text[start];
			Node** address = &(root->children[character]);
			Node* child = *address;
			if(child == NULL){
				child = createNode(start,end);
				addNode(address,child,text);
			} else {
				child = createNode(start,end);
				combineNode(address,child,text);
			}
		}
	}
}

void impl2(const char* text, int* indices, int* suffixes, 
		int totalLength, int numStrings, int numSuffixes, 
		int bsize, int bcount){
	Timer timer;
	Node root;
	root.start=0;
	root.end=0;
	for(int i = 0; i < NUM_CHILDREN; i++)
		root.children[i] = NULL;
	root.suffixIndex = 0;

	char* d_text = NULL;
        int* d_indices = NULL;
	int* d_suffixes = NULL;
	Node* d_root = NULL;

	cudaMalloc((void**)&d_text, sizeof(char)*totalLength);
        cudaMalloc((void**)&d_indices, sizeof(int)*numStrings);
	cudaMalloc((void**)&d_suffixes,sizeof(int)*numSuffixes);
        cudaMalloc((void**)&d_root, sizeof(Node));

        cudaMemcpy(d_text, text, sizeof(char)*totalLength, cudaMemcpyHostToDevice);
        cudaMemcpy(d_indices, indices, sizeof(int)*numStrings, cudaMemcpyHostToDevice);
        cudaMemcpy(d_suffixes, suffixes, sizeof(int)*numSuffixes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_root,&root,sizeof(Node),cudaMemcpyHostToDevice);

	timer.set();

	constructSuffixTree<<<bcount,bsize>>>(d_root,
		d_text,d_indices,d_suffixes,
		totalLength,numStrings,numSuffixes);
	cudaDeviceSynchronize();

	cout << "running time: " << timer.get() << " ms" << endl;

	printTree<<<1,1>>>(d_root,d_text);
	cudaDeviceSynchronize();

	// free
	cudaFree(d_text);
	cudaFree(d_indices);
	cudaFree(d_suffixes);
	cudaFree(d_root);
}
