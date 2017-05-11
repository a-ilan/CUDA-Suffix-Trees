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
	return "ERROR: Input may only contain alphanumeric characters.";
}

void toLowercase(string* data){
	transform(data->begin(), data->end(), data->begin(), ::tolower);
}

void saveResults(ofstream& outFile, char* solution){
	outFile << solution << endl;
}


vector<string> parseFile(ifstream& inFile){
	string line;
	vector<string> result;
	while(getline(inFile,line)){
		toLowercase(&line);
		result.push_back(line);
	}
	return result;
}

void parseStrings(vector<string> strings,
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
	for(int i = 0; i < strings.size(); i++){
		line = strings[i];
		stringstream ss;
		ss << line << "#" << *numStrings << "$";
		strings_result += ss.str();
		indices_result.push_back(*totalLength);
		for(int i = 0; i < line.length(); i++){
			suffixes_result.push_back((*totalLength)+i);
			if(!isalnum(line[i])) {
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

