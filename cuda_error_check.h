#ifndef	CUDAERRCHCK_H
#define	CUDAERRCHCK_H

#include <cuda_runtime.h>
#include <string>
#include <sstream>
#include <stdexcept>

//Error checking mechanism
#define CUDAErrorCheck(err) { CUDAAssert((err), __FILE__, __LINE__); }
inline void CUDAAssert( cudaError_t err, const char *file, int line )
{
   if ( err != cudaSuccess )
   {
	  std::ostringstream errStream;
	  errStream << "CUDAAssert: " << file << "(" << line << "): " << cudaGetErrorString(err);
      throw std::runtime_error( errStream.str() );
   }
}

#endif	//	CUDAERRCHCK_CUH
