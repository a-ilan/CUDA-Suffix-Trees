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

//parse a file
//for example input with the following words: apple, banana, cat, dog
//text: "apple#0$banana#1$cat#2$dog#3$" 
//    where '#' specifies the start of a terminating sequence
//    and '$' specifies the end of a terminating sequence
//indices: [0, 8, 17, 23]
//suffixIndices: [0,1,2,3,4, 8,9,10,11,12,13, 17,18,19, 23,24,25]
void parse_file(ifstream* inFile, 
		const char** text, int** indices, int** suffixIndices,  
		int* totalLength, int* numStrings, int* numSuffixes){
	string line;
	string* strings_result = new string();
	vector<int>* indices_result = new vector<int>();
	vector<int>* suffixes_result = new vector<int>();
	*numStrings = 0;
	*totalLength = 0;
	*numSuffixes = 0;
	while(getline(*inFile,line)){
		stringstream ss;
		ss << line << "#" << *numStrings << "$";
		*strings_result += ss.str();
		indices_result->push_back(*totalLength);
		for(int i = 0; i < line.length(); i++){
			suffixes_result->push_back((*totalLength)+i);
		}
		(*numStrings)++;
		(*totalLength) += ss.str().length();
		(*numSuffixes) += line.length();
	}

	*text = strings_result->c_str();
	*indices = &indices_result->at(0);
	*suffixIndices = &suffixes_result->at(0);
}
