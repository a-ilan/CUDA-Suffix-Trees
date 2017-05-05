#include <iostream>
#include <fstream>
#include <vector>
#include "cuda_error_check.h"
#include "implementation.h"
#include "utils.h"

using namespace std;

int main(int argc, char** argv){
	try {
		//declare and initialize variabls
		string usage =
		"\tCommand line arguments:\n\
                        Input file: E.g., --input in.txt\n\
                        Output path: E.g., --output out.txt\n\
                        Block size: E.g., --bsize 1024\n\
                        Block count: E.g., --bcount 2\n\
                        Method: E.g., --method 1, 2, or 3\n";
		string inputFileName;
		string outputFileName;
		ifstream inputFile;
		ofstream outputFile;
		int bsize = 0, bcount = 0;
		int method = 0;
		int device_id = 0;
		cudaDeviceProp device_prop;
		char* device_name = NULL;

		//check that CUDA is supported and get the name of the device
		CUDAErrorCheck(cudaSetDevice(device_id));
		CUDAErrorCheck(cudaGetDeviceProperties(&device_prop, device_id));
		device_name = device_prop.name;
	
		//parse program arguments
		for( int i = 1; i < argc; i++ ){
			if ( !strcmp(argv[i], "--input") && i != argc-1 ) {
				inputFileName = string(argv[i+1]);
				inputFile.open(inputFileName.c_str());
			} else if( !strcmp(argv[i], "--output") && i != argc-1 ) {
				outputFileName = string(argv[i+1]);
				outputFile.open(outputFileName.c_str());
			} else if( !strcmp(argv[i], "--bsize") && i != argc-1 ) {
				bsize = atoi( argv[i+1] );
			} else if( !strcmp(argv[i], "--bcount") && i != argc-1 ) {
				bcount = atoi( argv[i+1] );
			} else if( !strcmp(argv[i], "--method") && i != argc-1 ) {
				method = atoi( argv[i+1] );
			}
		}

		//verify program arguments
		if(!inputFile){
			throw runtime_error("Failed to open specified file: " + inputFileName);
		}
		if(!outputFile){
			throw runtime_error("Failed to open specified file: " + outputFileName);
		}
		if(!inputFile.is_open() || !outputFile.is_open()){
			cerr << "Usage: " << usage;
			throw runtime_error("Initialization error happened: input/output file");
		}
		if(bsize <= 0 || bcount <= 0){
			cerr << "Usage: " << usage;
			throw runtime_error("Initialization error happened: block size/count");
		}
		if(method == 0){
			cerr << "Usage: " << usage;
			throw runtime_error("Initialization error happened: method");
		}

		//parse input file
		const char* strings; //all the strings concatenated into a single string
		int* indices; //the starting index of each string
		int totalLength; //total length of all the strings
		int numStrings; //number of strings
		parse_file(&inputFile,&strings,&indices,&totalLength,&numStrings);
		inputFile.close();

		//print program properties
		cout << "Device: " << device_name;
		cout << ", bsize: " << bsize << ", bcount: " << bcount;
		cout << ", method: " << method << endl;	
		cout << "Input file: " << inputFileName;
		cout << ", Number of strings: " << numStrings;
		cout << ", total length: " << totalLength << endl;

		//process method
		switch(method){
		case 1:
			impl1(strings, indices, totalLength, numStrings, bsize, bcount);
			break;
		case 2:
			impl2(strings, indices, totalLength, numStrings, bsize, bcount);
			break;
		case 3:
			impl3(strings, indices, totalLength, numStrings, bsize, bcount);
			break;
		default:
			cout << "Method " << method << " does not exist. Try method 1, 2, or 3.\n";
			break;
		}

		//clean program memory
		CUDAErrorCheck(cudaDeviceReset());
		outputFile.close();

	} catch(const exception& e){
		cerr << e.what() << endl;
		return EXIT_FAILURE;
	} catch(...) {
		cerr << "An exception has occurred." << endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
