#ifndef IMPLEMENTATION_H
#define IMPLEMENTATION_H

#include <iostream>
#include <vector>
#include "utils.h"
using namespace std;

#define NUM_CHILDREN 256
typedef unsigned long long int address_type;

//Node struct
struct Node{
	//this is a compressed suffix tree,
	//which means an edge can have multiple characters.
	//none branching nodes are compressed into a single node

	//(start,end) interval specifies the characters of the edge
	//from the parent node to the current node.
	//Let's say there are two nodes A and B
	//connected by an edge with indices (5,8) then these indices
	//(5,8) will be stored in node B.
	//start and end are the starting and ending position of
	//a substring in "text"
	int start;
	int end;

	//children nodes
	Node* children[NUM_CHILDREN];

	//for leaf nodes, it stores the "index" of the suffix
	//index is the starting poisition of the suffix in "text"
	int suffixIndex;
};

// function prototype in impl1.h, impl2.h, and impl3.h
void impl1(const char* text, int* indices, 
	int totalLength, int numStrings, 
	int bsize, int bcount);
void impl2(const char* text, int* indices, int* suffixes,
	int totalLength, int numStrings, int numSuffixes,
	int bsize, int bcount);
void impl3(const char* text, int* indices, 
	int totalLength, int numStrings, 
	int bsize, int bcount);

// Node functions in impl_util.cu
__device__ Node* createNode(int start, int end);
__device__ bool splitNode(struct Node** address, int position, const char* text);
__device__ void combineNode(struct Node** address, struct Node* node2, const char* text);
__device__ void addNode(struct Node** address, struct Node* node2, const char* text);
__device__ void printNode(struct Node* node, const char* text);
__global__ void printTree(struct Node* root, const char* text);

#endif
