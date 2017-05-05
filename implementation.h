#ifndef IMPLEMENTATION_H
#define IMPLEMENTATION_H

#include <iostream>
#include <vector>
#include "utils.h"
using namespace std;

struct Node{
	Node* children[256];
};

void impl1(const char* strings, int* indices, int totalLength, int numStrings, int bsize, int bcount);
void impl2(const char* strings, int* indices, int totalLength, int numStrings, int bsize, int bcount);
void impl3(const char* strings, int* indices, int totalLength, int numStrings, int bsize, int bcount);

#endif
