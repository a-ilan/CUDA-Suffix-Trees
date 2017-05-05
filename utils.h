#ifndef UTILS_H
#define UTILS_H

#include <iostream>
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

void parse_file(ifstream* inFile, const char** strings, int** indices, int* totalLength, int* numStrings);

#endif
