#include "implementation.h"

__global__ void constructSuffixTree(Node* root, char* text, int* indices, int totalLength, int numStrings){
	const int tid = threadIdx.x + blockDim.x*blockIdx.x;
	const int nThreads = blockDim.x*gridDim.x;
	const int iter = numStrings%nThreads == 0? numStrings/nThreads : numStrings/nThreads+1;

	for(int i = 0; i < iter; i++){
		int dataid = tid + i*nThreads;
		if(dataid < numStrings){
			int start = indices[dataid];
			int end = dataid == numStrings-1? totalLength : indices[dataid+1];
			for(; text[start] != '#'; start++){
				char c = text[start];
				char index = charToIndex(c);
				Node** address = &(root->children[index]);
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
}

__global__ void constructSuffixTree2(Node* root, char* text, int* indices, int totalLength, int start_index, int end_index){
	const int tid = threadIdx.x + blockDim.x*blockIdx.x;
	const int nThreads = blockDim.x*gridDim.x;
	int numStrings = end_index-start_index;
	const int iter = numStrings%nThreads == 0? numStrings/nThreads : numStrings/nThreads+1;

	for(int i = 0; i < iter; i++){
		int dataid = start_index + tid + i*nThreads;
		if(dataid < end_index){
			int start = indices[dataid];
			int end = dataid == numStrings-1? totalLength : indices[dataid+1];
			for(; text[start] != '#'; start++){
				char c = text[start];
				char index = charToIndex(c);
				Node** address = &(root->children[index]);
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
}

char* impl1_batches(char* text, int* indices, int totalLength, int numStrings, int bsize, int bcount){
	Timer timer;
	Node root;
	root.start=0;
	root.end=0;
	for(int i = 0; i < NUM_CHILDREN; i++)
		root.children[i] = NULL;
	root.suffixIndex = 0;

        char* d_text = NULL;
        int* d_indices = NULL;
	Node* d_root = NULL;

        cudaMalloc((void**)&d_text, sizeof(char)*totalLength);
        cudaMalloc((void**)&d_indices, sizeof(int)*numStrings);
        cudaMalloc((void**)&d_root, sizeof(Node));

        cudaMemcpy(d_text, text, sizeof(char)*totalLength, cudaMemcpyHostToDevice);
        cudaMemcpy(d_indices, indices, sizeof(int)*numStrings, cudaMemcpyHostToDevice);
	cudaMemcpy(d_root,&root,sizeof(Node),cudaMemcpyHostToDevice);

	timer.set();

	int stringsPerBatch = bcount*bsize;
	int numBatches = numStrings % stringsPerBatch == 0? numStrings/stringsPerBatch : numStrings/stringsPerBatch+1;

	for(int i = 0; i < numBatches; i++){
		int start = i*stringsPerBatch;
		int end = min(start+stringsPerBatch,numStrings);
		constructSuffixTree2<<<bcount,bsize>>>(d_root,d_text,d_indices,totalLength,start,end);
		CUDAErrorCheck(cudaPeekAtLastError());
		CUDAErrorCheck(cudaDeviceSynchronize());
	}
	
	cout << "running time: " << timer.get() << " ms" << endl;

	//printTree<<<1,1>>>(d_root,d_text);
	//cudaDeviceSynchronize();

	char* output = NULL;
	int size = getSerialSuffixTree(d_root,d_text,&output);
	printf("Output size: %d\n",size);

	// free
	cudaFree(d_text);
	cudaFree(d_indices);
	cudaFree(d_root);

	return output;
}

char* impl1(char* text, int* indices, int totalLength, int numStrings, int bsize, int bcount){
	Timer timer;
	Node root;
	root.start=0;
	root.end=0;
	for(int i = 0; i < NUM_CHILDREN; i++)
		root.children[i] = NULL;
	root.suffixIndex = 0;

        char* d_text = NULL;
        int* d_indices = NULL;
	Node* d_root = NULL;

        cudaMalloc((void**)&d_text, sizeof(char)*totalLength);
        cudaMalloc((void**)&d_indices, sizeof(int)*numStrings);
        cudaMalloc((void**)&d_root, sizeof(Node));

        cudaMemcpy(d_text, text, sizeof(char)*totalLength, cudaMemcpyHostToDevice);
        cudaMemcpy(d_indices, indices, sizeof(int)*numStrings, cudaMemcpyHostToDevice);
	cudaMemcpy(d_root,&root,sizeof(Node),cudaMemcpyHostToDevice);

	timer.set();

	constructSuffixTree<<<bcount,bsize>>>(d_root,d_text,d_indices,totalLength,numStrings);

	CUDAErrorCheck(cudaPeekAtLastError());
	CUDAErrorCheck(cudaDeviceSynchronize());
	
	cout << "running time: " << timer.get() << " ms" << endl;

	//printTree<<<1,1>>>(d_root,d_text);
	//cudaDeviceSynchronize();

	char* output = NULL;
	int size = getSerialSuffixTree(d_root,d_text,&output);
	printf("Output size: %d\n",size);

	// free
	cudaFree(d_text);
	cudaFree(d_indices);
	cudaFree(d_root);

	return output;
}

/*
char* impl1_batches(char** text, int** indices, int* totalLength, int* numStrings, int numBatches, int bsize, int bcount){
	Timer timer;
	Node root;
	root.start=0;
	root.end=0;
	for(int i = 0; i < NUM_CHILDREN; i++)
		root.children[i] = NULL;
	root.suffixIndex = 0;

	Node* d_root = NULL;
        cudaMalloc((void**)&d_root, sizeof(Node));
	cudaMemcpy(d_root,&root,sizeof(Node),cudaMemcpyHostToDevice);
   
	char** d_text = NULL;
	int** d_indices = NULL;

	for(int i = 0; i < numBatches; i++){
        	cudaMalloc((void**)&(d_text[i]), sizeof(char)*totalLength[i]);
        	cudaMalloc((void**)&(d_indices[i]), sizeof(int)*numStrings[i]);

        	cudaMemcpy(d_text[i], text, sizeof(char)*totalLength[i], cudaMemcpyHostToDevice);
        	cudaMemcpy(d_indices[i], indices, sizeof(int)*numStrings[i], cudaMemcpyHostToDevice);
	}

	timer.set();

	for(int i = 0; i < numBatches; i++){
		constructSuffixTree<<<bcount,bsize>>>(d_root,d_text[i],d_indices[i],totalLength[i],numStrings[i]);
		CUDAErrorCheck(cudaPeekAtLastError());
		CUDAErrorCheck(cudaDeviceSynchronize());
	}
	
	cout << "running time: " << timer.get() << " ms" << endl;

	//printTree<<<1,1>>>(d_root,d_text);
	//cudaDeviceSynchronize();

	char* output = NULL;
	int size = getSerialSuffixTree(d_root,d_text,&output);
	printf("Output size: %d\n",size);

	// free
	cudaFree(d_text);
	cudaFree(d_indices);
	cudaFree(d_root);

	return output;
}*/
