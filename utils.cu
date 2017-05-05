#include "utils.h"

void Timer::set(){
	gettimeofday( &startingTime, NULL );
}

double Timer::get(){
	struct timeval pausingTime, elapsedTime;
	gettimeofday( &pausingTime, NULL );
	timersub(&pausingTime, &startingTime, &elapsedTime);
	return elapsedTime.tv_sec*1000.0+elapsedTime.tv_usec/1000.0;// Returning in milliseconds.
}

void parse_file(ifstream* inFile, const char** strings, int** indices, int* totalLength, int* numStrings){
        string line;
        string* strings_result = new string();
        vector<int>* indices_result = new vector<int>();
        *numStrings = 0;
        *totalLength = 0;
        while(getline(*inFile,line)){
                *strings_result += line;
                indices_result->push_back(*totalLength);
                (*numStrings)++;
                (*totalLength) += line.length();
        }

        *strings = strings_result->c_str();
        *indices = &indices_result->at(0);
}
