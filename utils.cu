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
		char*& text, int*& indices, int*& suffixes,  
		int& totalLength, int& numStrings, int& numSuffixes)
		throw(NotAllowedSymbolException){
	string line;
	string strings_result;
	vector<int> indices_result;
	vector<int> suffixes_result;
	numStrings = 0;
	totalLength = 0;
	numSuffixes = 0;
	for(int i = 0; i < strings.size(); i++){
		line = strings[i];
		stringstream ss;
		ss << line << "#" << numStrings << "$";
		strings_result += ss.str();
		indices_result.push_back(totalLength);
		for(int i = 0; i < line.length(); i++){
			suffixes_result.push_back((totalLength)+i);
			if(!isalnum(line[i])) {
				throw NotAllowedSymbolException();
			}
		}
		numStrings++;
		totalLength += ss.str().length();
		numSuffixes += line.length();
	}

	text = (char*)malloc(totalLength+1);
	indices = (int*)malloc(sizeof(int)*(numStrings));
	suffixes = (int*)malloc(sizeof(int)*(numSuffixes));

	strcpy(text,strings_result.c_str());
	memcpy(indices,&indices_result[0],sizeof(int)*(numStrings));
	memcpy(suffixes,&suffixes_result[0],sizeof(int)*(numSuffixes));
}

void parseToBatches(vector<string> strings, int numStringsPerBatch, int& numBatches,
		char**& text, int**& indices, int**& suffixes,  
		int*& totalLength, int*& numStrings, int*& numSuffixes)
		throw(NotAllowedSymbolException){
	numBatches = strings.size()%numStringsPerBatch==0? strings.size()/numStringsPerBatch : strings.size()/numStringsPerBatch+1;

	text = new char*[numBatches];
	indices = new int*[numBatches];
	suffixes = new int*[numBatches];
	totalLength = new int[numBatches];
	numStrings = new int[numBatches];
	numSuffixes = new int[numBatches];
	
	memset(totalLength, 0, sizeof(int)*numBatches);
	memset(numStrings, 0, sizeof(int)*numBatches);
	memset(numSuffixes, 0, sizeof(int)*numBatches);

	int start = 0;
	for(int j = 0; j < numBatches; j++){
		string line;
		string strings_result;
		vector<int> indices_result;
		vector<int> suffixes_result;
		numStrings[j] = 0;
		totalLength[j] = 0;
		numSuffixes[j] = 0;
		int end = min(start + numStringsPerBatch, (int)strings.size());
		for(int i = start; i < end; i++){
			line = strings[i];
			stringstream ss;
			ss << line << "#" << numStrings[j] << "$";
			strings_result += ss.str();
			indices_result.push_back(totalLength[j]);
			for(int i = 0; i < line.length(); i++){
				suffixes_result.push_back((totalLength[j])+i);
				if(!isalnum(line[i])) {
					throw NotAllowedSymbolException();
				}
			}
			numStrings[j]++;
			totalLength[j] += ss.str().length();
			numSuffixes[j] += line.length();
			start = end;
		}

		text[j] = new char[totalLength[j]+1];
		indices[j] = new int[numStrings[j]];
		suffixes[j] = new int[numSuffixes[j]];

		strcpy(text[j],strings_result.c_str());
		memcpy(indices[j],&indices_result[0],sizeof(int)*numStrings[j]);
		memcpy(suffixes[j],&suffixes_result[0],sizeof(int)*numSuffixes[j]);
	}
}


