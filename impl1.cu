#include "implementation.h"
typedef unsigned long long int address_type;

__device__ Node* createNode(int start, int end);
__device__ void splitNode(Node** address, int position, const char* text);
__device__ void combineNode(Node** address, Node* node2, const char* text);
__device__ void addNode(Node** address, Node* node2, const char* text);

__device__ Node* createNode(int start, int end){
	Node* node = (Node*)malloc(sizeof(Node));
	for(int i = 0; i < NUM_CHILDREN; i++)
		node->children[i] = NULL;
	node->start = start;
	node->end = end;
	return node;
}

//split a node into a parent and child node at a specified position
__device__ void splitNode(Node** address, int position, const char* text){
	//current node;
	Node* node = *address;

	//branchingNode is the child node of parentNode
	Node* branchingNode = (Node*)malloc(sizeof(Node));
	for(int i = 0; i < NUM_CHILDREN; i++){
		branchingNode->children[i] = node->children[i];
	}
	branchingNode->start = position;
	branchingNode->end = node->end;

	//parentNode is the parent of branchingNode
	Node* parentNode = (Node*)malloc(sizeof(Node));
	for(int i = 0; i < NUM_CHILDREN; i++){
		parentNode->children[i] = NULL;
	}
	parentNode->start = node->start;
	parentNode->end = position;
	char character = text[position];
	parentNode->children[character] = branchingNode;

	atomicCAS((address_type*)address, (address_type)node, (address_type)parentNode);
	if(*address != parentNode){
		//free parentNode and branchingNode
		free(parentNode);
		free(branchingNode);
	}
}

//combines the suffix of a node in the tree with another node
//address is the address of node1
//node1 is a node in the suffix tree
//node2 is a node that needs to be added to the suffix tree
__device__ void combineNode(Node** address, Node* node2, const char* text){
	Node* node1 = *address;
	int i = node1->start;
	int j = node2->start;
	int i_end = node1->end;
	int j_end = node2->end;
	while(i < i_end && j < j_end){
		if(text[i] != text[j]){
			splitNode(address,i,text); 
			node2->start = j;
			char c = text[j];
			Node** address2 = &((*address)->children[c]);
			addNode(address2,node2,text);
			break;
		}
		i++;
		j++;
	}
	if(i == i_end && j != j_end){
		node2->start = j;
		char c = text[j];
		Node** address2 = &((*address)->children[c]);
		addNode(address2,node2,text);
	}
	else if(i != i_end && j == j_end){
	}
}

//add a child node to a node in the tree (*address)
__device__ void addNode(Node** address, Node* child, const char* text){
	  //address = address==NULL? child : address;
	atomicCAS((address_type*)address, NULL, (address_type)child); //set address = child
	if(child != *address){ //check if atomicCAS failed
		combineNode(address,child,text);
	}
}

__global__ void construct_suffix_tree(Node* root, const char* text, int* indices, int totalLength, int numStrings){
	const int tid = threadIdx.x + blockDim.x*blockIdx.x;
	const int nThreads = blockDim.x*gridDim.x;
	const int iter = totalLength%nThreads == 0? totalLength/nThreads : totalLength/nThreads+1;

	for(int i = 0; i < iter; i++){
		int dataid = tid + i*nThreads;
		if(dataid < numStrings){
			int start = indices[dataid];
			int end = dataid == numStrings-1? totalLength : indices[dataid+1];
			//for(; text[start] != '#'; start++){
			char character = text[start];
			Node** address = &(root->children[character]);
			Node* child = *address;
			if(child == NULL){
				child = createNode(start,end);
				addNode(address,child,text);
			} else {
				combineNode(address,child,text);
			}
		}
	}
}


__device__ int getNodeString(char* buf, Node* node, const char* text){
	int start = node->start;
	int end = node->end;
	int i = 0;
	for(int j = start; j < end; j++){
		buf[i] = text[j];
		i++;
	}
	buf[i] = '\0';
	return i;
}

__device__ void printTree(Node* root, const char* text, int indent){
	for(int i = 0; i < NUM_CHILDREN; i++){
		Node* child = root->children[i];
		if(child != NULL){
			char buf[255];
			int size = getNodeString(buf,child,text);
			printf("%*s\n",size+indent, buf);
			printTree(child,text,indent+1);
		}
	}
}

__global__ void print_tree(Node* root, const char* text){
	printTree(root,text,0);
}

void impl1(const char* text, int* indices, int totalLength, int numStrings, int bsize, int bcount){
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

	construct_suffix_tree<<<bcount,bsize>>>(d_root,d_text,d_indices,totalLength,numStrings);
	//construct_suffix_tree<<<1,1>>>(d_root,d_text,d_indices,totalLength,numStrings);
	
	print_tree<<<1,1>>>(d_root,d_text);

	cout << "running time: " << timer.get() << " ms" << endl;

	// free
	cudaFree(d_text);
	cudaFree(d_indices);
	cudaFree(d_root);
}
