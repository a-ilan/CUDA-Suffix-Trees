#include "utils.h"
#include <string>
#include <algorithm>

void Timer::set(){
	gettimeofday( &startingTime, NULL );
}

double Timer::get(){
	struct timeval pausingTime, elapsedTime;
	gettimeofday( &pausingTime, NULL );
	timersub(&pausingTime, &startingTime, &elapsedTime);
	return elapsedTime.tv_sec*1000.0+elapsedTime.tv_usec/1000.0;// Returning in milliseconds.
}

const char* NotAllowedSymbolException::what() const throw(){
	return "Input contains non-allowed symbols ('$', '#', and ')').";
}

void toLowercase(string* data){
	transform(data->begin(), data->end(), data->begin(), ::tolower);
}

//parse a file
//for example input with the following words: apple, banana, cat, dog
//text: "apple#0$banana#1$cat#2$dog#3$" 
//    where '#' specifies the start of a terminating sequence
//    and '$' specifies the end of a terminating sequence
//indices: [0, 8, 17, 23]
//suffixIndices: [0,1,2,3,4, 8,9,10,11,12,13, 17,18,19, 23,24,25]
void parseFile(ifstream* inFile, 
		char** text, int** indices, int** suffixes,  
		int* totalLength, int* numStrings, int* numSuffixes)
		throw(NotAllowedSymbolException){
	string line;
	string strings_result;
	vector<int> indices_result;
	vector<int> suffixes_result;
	*numStrings = 0;
	*totalLength = 0;
	*numSuffixes = 0;
	while(getline(*inFile,line)){
		stringstream ss;
		toLowercase(&line);
		ss << line << "#" << *numStrings << "$";
		strings_result += ss.str();
		indices_result.push_back(*totalLength);
		for(int i = 0; i < line.length(); i++){
			suffixes_result.push_back((*totalLength)+i);
			if(line[i] == '$' || line[i] == '#' ||
					line[i] == ')') {
				throw NotAllowedSymbolException();
			}
		}
		(*numStrings)++;
		(*totalLength) += ss.str().length();
		(*numSuffixes) += line.length();
	}

	*text = (char*)malloc(*totalLength+1);
	*indices = (int*)malloc(sizeof(int)*(*numStrings));
	*suffixes = (int*)malloc(sizeof(int)*(*numSuffixes));

	strcpy(*text,strings_result.c_str());
	memcpy(*indices,&indices_result[0],sizeof(int)*(*numStrings));
	memcpy(*suffixes,&suffixes_result[0],sizeof(int)*(*numSuffixes));
}

