#include "implementation.h"

__device__ Node* createNode(int start, int end){
	Node* node = (Node*)malloc(sizeof(Node));
	for(int i = 0; i < NUM_CHILDREN; i++)
		node->children[i] = NULL;
	node->start = start;
	node->end = end;
	return node;
}

//split a node into a parent and child node at a specified position
//return true if node successfully split
__device__ bool splitNode(Node** address, int position, char* text){
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
		return false;
	}
	return true;
}

//combines the suffix of a node in the tree with another node
//address is the address of node1
//node1 is a node in the suffix tree
//node2 is a node that needs to be added to the suffix tree
__device__ void combineNode(Node** address, Node* node2, char* text){
	Node* node1 = *address;
	int i = node1->start;
	int j = node2->start;
	int i_end = node1->end;
	int j_end = node2->end;
	while(i < i_end && j < j_end){
		if(text[i] != text[j]){
			bool isSplit = splitNode(address,i,text); 
			if(isSplit){
				node2->start = j;
				char c = text[j];
				Node** address2 = &((*address)->children[c]);
				addNode(address2,node2,text);
			}
			else{
				//failed to split because another node already caused it split
				//try again
				combineNode(address,node2,text);
			}
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
	else {
	}
}

//add a child node to a node in the tree (*address)
__device__ void addNode(Node** address, Node* child, char* text){
	  //address = address==NULL? child : address;
	atomicCAS((address_type*)address, NULL, (address_type)child); //set address = child
	if(child != *address){ //check if atomicCAS failed
		combineNode(address,child,text);
	}
}

__device__ int nodeToString(char* buf, Node* node, char* text){
	int start = node->start;
	int end = node->end;
	int i = 0;
	for(int j = start; j < end; j++){
		if(i == 19) break;
		buf[i] = text[j];
		i++;
	}
	buf[i] = '\0';
	return i;
}

__device__ void printNode(Node* node, char* text){
	char buf[20];
	nodeToString(buf,node,text);
	printf("%s\n",buf);
}

__device__ void printTreeRe(Node* root, char* text, int indent){
	for(int i = 0; i < NUM_CHILDREN; i++){
		Node* child = root->children[i];
		if(child != NULL){
			char buf[20];
			int size = nodeToString(buf,child,text);
			printf("%*s\n",indent+size, buf);
			printTreeRe(child,text,indent+1);
		}
	}
}

__global__ void printTree(Node* root, char* text){
	printf("PRINTING TREE:\n");
	printTreeRe(root,text,0);
}

// count characters
__host__ __device__ void countChar(Node* root, int* numChar){
    if(root == NULL){
	return;
    }
    int num = root->end - root->start;
    *numChar += num;
    for(int i = 0 ; i < NUM_CHILDREN ; i++){
	countChar(root->children[i], numChar);
    }
    (*numChar)+= num;
}

// pre-order traversal
__host__ __device__ void serialize(Node* root, char* text, char* output, int* counter){
    // base case
    if(root == NULL) return;
    // number of characters in node
    int num = root->end - root->start;
    // copy characters
    for(int i = 0; i < num ; i++){
	output[*counter+i] = text[root->start + i];
    }
    *counter += num;
    for(int i = 0 ; i < NUM_CHILDREN; i++){
	serialize(root->children[i], text, output, counter);
    }

    for(int i = 0; i < num; i++){
	output[*counter] = ')';
	(*counter)++;
    }
}

__global__ void countCharForSerialization(Node* root, int* numChar){
	*numChar = 1; // one for last char to be null
	countChar(root,numChar);
}

__global__ void serialization(Node* root, char* text, char* output){
	int counter = 0;
	serialize(root,text,output,&counter);
	output[counter] = '\0';
}

//copy suffix tree from device to host by serializing it
//d_root should be pointer to an address in the device
//d_text should be pointer to an address in the device
//output shoud be the reference to the char* where you want the output to be
//return the size of output
int getSerialSuffixTree(Node* d_root, char* d_text, char** output){
	int size = 0;
	int* d_size = NULL;
	char* d_output = NULL;	
	cudaMalloc((void**)&d_size,sizeof(int));

	countCharForSerialization<<<1,1>>>(d_root,d_size);
	cudaDeviceSynchronize();
	cudaMemcpy(&size,d_size,sizeof(int),cudaMemcpyDeviceToHost);
	cudaMalloc((void**)&d_output,size);
	*output = (char*)malloc(size);
	serialization<<<1,1>>>(d_root,d_text,d_output);
	cudaMemcpy(*output,d_output,size,cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	cudaFree(d_output);
	cudaFree(d_size);
	return size;
}
