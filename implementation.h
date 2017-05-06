#ifndef IMPLEMENTATION_H
#define IMPLEMENTATION_H

#include <iostream>
#include <vector>
#include "utils.h"
using namespace std;

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
	Node* children[256];

	//for leaf nodes, it stores the "index" of the suffix
	//index is the starting poisition of the suffix in "text"
	int suffixIndex;
};

// function prototype
void impl1(const char* text, int* indices, int totalLength, int numStrings, int bsize, int bcount);
void impl2(const char* text, int* indices, int totalLength, int numStrings, int bsize, int bcount);
void impl3(const char* text, int* indices, int totalLength, int numStrings, int bsize, int bcount);

#endif
