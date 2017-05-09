#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <exception>
#include <sstream>
#include <fstream>
#include <vector>
#include <sys/time.h>
using namespace std;


class Timer {
	struct timeval startingTime;
public:
	void set();
	double get();
};

class NotAllowedSymbolException: public exception{
public:
	virtual const char* what() const throw();
};

void parseFile(ifstream* inFile, 
		const char** text, int** indices, int** suffixIndices,  
		int* totalLength, int* numStrings, int* numSuffixes) 
		throw (NotAllowedSymbolException);

// count characters
__global__ __host__ countChar(Node* root, const char* text, int* numChar);

// pre-order traversal
__global__ __host__ serialize(Node* root, const char* text, char* output, int* numChar);

#endif

