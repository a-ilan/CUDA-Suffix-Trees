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
__device__ bool splitNode(Node** address, int position, const char* text){
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
__device__ void combineNode(Node** address, Node* node2, const char* text){
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
__device__ void addNode(Node** address, Node* child, const char* text){
	  //address = address==NULL? child : address;
	atomicCAS((address_type*)address, NULL, (address_type)child); //set address = child
	if(child != *address){ //check if atomicCAS failed
		combineNode(address,child,text);
	}
}

__device__ int nodeToString(char* buf, Node* node, const char* text){
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

__device__ void printNode(Node* node,const char* text){
	char buf[20];
	nodeToString(buf,node,text);
	printf("%s\n",buf);
}

__device__ void printTreeRe(Node* root, const char* text, int indent){
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

__global__ void printTree(Node* root, const char* text){
	printTreeRe(root,text,0);
}

