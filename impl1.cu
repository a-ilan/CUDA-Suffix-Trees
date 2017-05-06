#include "implementation.h"
typedef unsigned long long int address_type;

__device__ Node* createNode(int start, int end){
	Node* node = (Node*)malloc(sizeof(Node));
	for(int i = 0; i < NUM_CHILDREN; i++)
		node->children[i] = NULL;
	node->start = start;
	node->end = end;
	return node;
}

__device__ Node* createNode(const char* text, int dataid){
	Node* node = (Node*)malloc(sizeof(Node));
	for(int i = 0; i < NUM_CHILDREN; i++)
		node->children[i] = NULL;
	node->start = dataid;
	int curr = node->start;
	while(text[curr] != '$') curr++;
	node->end = curr;
	return node;
}

__device__ Node* copyNode(Node* other){
	Node* node = (Node*)malloc(sizeof(Node));
	for(int i = 0; i < NUM_CHILDREN; i++)
		node->children[i] = other->children[i];
	node->start = other->start;
	node->end = other->end;
	return node;
}

//atomically create a child node from parent at childIndex
__device__ Node* atomicCreateChildNode(Node* parent, int childIndex, const char* text, int dataid){
	  //address of the child
	Node** address = &(parent->children[childIndex]);
	Node* child = *address;
	if(child == NULL){
		child = createNode(text,dataid);
		  //address = address==NULL? child : address;
		atomicCAS((address_type*)address, NULL, (address_type)child); //set address = child
		if(child != *address){ //check if atomicCAS failed
			//if it failed it means another thread already created the node
			//so so the edge need to branch into 2 different nodes (child1,child2) 
			Node* child1 = child; 
			Node* child2 = copyNode(*address);
			Node* parent = *address;

			//combine child and child2
			int child1_curr = child1->start;
			int child2_curr = child2->start;
			while(text[child1_curr] != '$' && text[child2_curr] != '$'){
				if(text[child1_curr] != text[child2_curr]){
					child1->start = child1_curr;
					child2->start = child2_curr;

					//cut the edge
					int minEnd = atomicMin(&parent->end, child2_curr);
					if(minEnd != child2->end){
						//another thread has cut the edge
						
					} else {
						atomicCreateChildNode(parent,child2_curr,text,dataid);
						atomicCreateChildNode(parent,child1_curr,text,dataid);
					}
					break;
				}
				child1_curr++;
				child2_curr++;
			}
		}
	}
	return child;
}

__device__ void addNode(Node** address, Node* node2, const char* text);
__device__ void getNodeString(char* buf, Node* node, const char* text);

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

	char buf1[255], buf2[255];
	getNodeString(buf1,parentNode,text);
	getNodeString(buf2,branchingNode,text);
	printf("parentNode %s, branchingNode %s\n",buf1,buf2);

	atomicCAS((address_type*)address, (address_type)node, (address_type)parentNode);
	if(*address != parentNode){
		//free parentNode and branchingNode
		free(parentNode);
		free(branchingNode);
		//add node to *address
		addNode(address,node,text);
	}
}

//address is the address of node1
//node1 is a node in the suffix tree
//node2 is a node that needs to be added to the suffix tree
__device__ void addNode(Node** address, Node* node2, const char* text){
	Node* node1 = *address;
	int i = node1->start;
	int j = node2->start;
	while(text[i] != '$' && text[j] != '$'){
		if(text[i] != text[j]){
			splitNode(address,i,text);
			//add node 2
			break;
		}
		i++;
		j++;
	}
	if(text[i] == '$' && text[j] != '$'){

	}
	else if(text[i] != '$' && text[j] == '$'){

	}
}

__device__ void addSuffix(Node* parent, int start, int end, const char* text){
	char character = text[start];
	Node** address = &(parent->children[character]);
	Node* child = *address;
	if(child == NULL){
		child = createNode(start,end);
		  //address = address==NULL? child : address;
		atomicCAS((address_type*)address, NULL, (address_type)child); //set address = child
		if(child != *address){ //check if atomicCAS failed
			addNode(address,child,text);
		}
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
			addSuffix(root,start,end,text);

			//char character = text[start];
			//if(character == '$') return;
			//Node* child = atomicCreateChildNode(root,character,text,dataid);
			//Node* child = root->children[character];
		}
	}
}


__device__ void getNodeString(char* buf, Node* node, const char* text){
	int start = node->start;
	int end = node->end;
	int i = 0;
	for(int j = start; j < end; j++){
		buf[i] = text[j];
		i++;
	}
	buf[i] = '\0';
}

__global__ void serialize_suffix_tree(Node* root, const char* text){
	for(int i = 0; i < NUM_CHILDREN; i++){
		if(root->children[i] != NULL){
			char buf[255];
			getNodeString(buf,root->children[i],text);
			printf("%c - %s\n",(char)i,buf);
			for(int j = 0; j < NUM_CHILDREN; j++){
				if(root->children[i]->children[j] != NULL){
					char buf[255];
					getNodeString(buf,root->children[i]->children[j],text);
					printf("  %c - %s\n",(char)j,buf);
				}
			}
		}
	}
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
	
	serialize_suffix_tree<<<1,1>>>(d_root,d_text);

	cout << "running time: " << timer.get() << " ms" << endl;

	// free
	cudaFree(d_text);
	cudaFree(d_indices);
	cudaFree(d_root);
}
