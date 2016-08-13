#include <pycuda-helpers.hpp>

extern "C"   // ensure functions name to be exactly the same as below
{
	__global__ void convertToUCHAR( const int field, const int nCells, double normaliztion, double *values, unsigned char *psiUCHAR ){
		int tid = blockIdx.x*blockDim.x + threadIdx.x;
		psiUCHAR[tid] = (unsigned char) ( -255*( values[field*nCells + tid]*normaliztion -1 ));
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	__global__ void reduction_max_kernel( double *input, double *output ){
		__shared__ double sh_data[512];

		unsigned int tid = threadIdx.x;
		unsigned int i   = blockIdx.x * blockDim.x  + threadIdx.x;
		sh_data[tid] = max( input[i], input[i + blockDim.x*gridDim.x ] ) ;
		__syncthreads();

		for( unsigned int s = blockDim.x/2; s>0; s >>= 1){
			if ( tid < s ) sh_data[tid] = max( sh_data[tid], sh_data[tid+s] );
			__syncthreads();
		}

		if ( tid == 0 ) output[ blockIdx.x ] = sh_data[0];
	}
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	__global__ void reduction_min_kernel( double *input, double *output ){
		__shared__ double sh_data[512];

		unsigned int tid = threadIdx.x;
		unsigned int i   = blockIdx.x * blockDim.x  + threadIdx.x;
		sh_data[tid] = min( input[i], input[i + blockDim.x*gridDim.x ] ) ;
		__syncthreads();

		for( unsigned int s = blockDim.x/2; s>0; s >>= 1){
			if ( tid < s ) sh_data[tid] = min( sh_data[tid], sh_data[tid+s] );
			__syncthreads();
		}

		if ( tid == 0 ) output[ blockIdx.x ] = sh_data[0];
	}
}
