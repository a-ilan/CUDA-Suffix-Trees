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

void saveResults(ofstream& outFile, char* solution);

void parseFile(ifstream* inFile, 
		char** text, int** indices, int** suffixIndices,  
		int* totalLength, int* numStrings, int* numSuffixes) 
		throw (NotAllowedSymbolException);


#endif

