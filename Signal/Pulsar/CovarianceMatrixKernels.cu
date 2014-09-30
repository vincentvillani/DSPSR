/*
 * CovarianceMatrixKernels.cu
 *
 *  Created on: 23/09/2014
 *      Author: vincentvillani
 */

#include "dsp/CovarianceMatrixKernels.h"


__global__ void outerProductKernel(float* result, unsigned int resultLength, float* vec, unsigned int vecLength)
{
	for(unsigned int absoluteThreadIdx = blockDim.x * blockIdx.x + threadIdx.x; absoluteThreadIdx < resultLength; absoluteThreadIdx += gridDim.x * blockDim.x)
	{
		unsigned int row = absoluteThreadIdx / vecLength;
		unsigned int col = absoluteThreadIdx % vecLength;

		if(row > col)
		{
			row = vecLength - row;
			col = row + col;
		}

		//compute the index
		int index = (row * vecLength + col) - ((row * (row + 1)) / 2);

		//do the outer product calculation and add it too the correct element
		result[index] += vec[row] * vec[col];

	}
}



//(d_amps, ampsLength, d_hits, stokesLength)
__global__ void meanStokesKernel(float* d_amps, unsigned int ampsLength, const unsigned int* d_hits, unsigned int stokesLength)
{
	unsigned int absoluteThreadIdx = blockDim.x * blockIdx.x + threadIdx.x;

	if(absoluteThreadIdx >= ampsLength)
		return;

	unsigned int hitVal = d_hits[ absoluteThreadIdx / stokesLength ];

	d_amps[absoluteThreadIdx] /= hitVal;

}



__global__ void applyScaleKernel(float* amps, unsigned int ampsLength, double scaleFactor)
{
	unsigned int absoluteThreadIdx = blockDim.x * blockIdx.x + threadIdx.x;

	if(absoluteThreadIdx >= ampsLength)
		return;

	amps[absoluteThreadIdx] /= scaleFactor;
}



//----PHASE SERIES COMBINE STUFF----


//Kernel for generically adding things on the GPU
__global__ void genericAddKernel(unsigned int n, float* original, const float* add)
{
	for(unsigned int absIdx = blockDim.x * blockIdx.x + threadIdx.x; absIdx < n; absIdx += gridDim.x * blockDim.x)
	{
		original[absIdx] += add[absIdx];
	}
}


//Kernel for generically adding things on the GPU
__global__ void genericAddKernel(unsigned long long int n, float* original, const float* add)
{
	for(unsigned int absIdx = blockDim.x * blockIdx.x + threadIdx.x; absIdx < n; absIdx += gridDim.x * blockDim.x)
	{
		original[absIdx] += add[absIdx];
	}
}



//Kernel for generically adding things on the GPU
__global__ void genericAddKernel(unsigned int n, unsigned int* original, const unsigned int* add)
{
	for(unsigned int absIdx = blockDim.x * blockIdx.x + threadIdx.x; absIdx < n; absIdx += gridDim.x * blockDim.x)
	{
		original[absIdx] += add[absIdx];
		//printf("AddVal: %u\n", add[absIdx]);
	}
}



__global__ void genericSubtractionKernel(unsigned int n, float* original, const float* sub)
{
	for(unsigned int absIdx = blockDim.x * blockIdx.x + threadIdx.x; absIdx < n; absIdx += gridDim.x * blockDim.x)
	{
		original[absIdx] -= sub[absIdx];
	}
}


__global__ void genericDivideKernel(unsigned int n, float* d_numerators, unsigned int denominator)
{
	for(unsigned int absIdx = blockDim.x * blockIdx.x + threadIdx.x; absIdx < n; absIdx += gridDim.x * blockDim.x)
	{
		d_numerators[absIdx] /= denominator;
	}
}



__global__ void checkForZeroesKernel(const unsigned int* d_hits, unsigned int hitsLength, bool* d_zeroes)
{
	for(unsigned int absIdx = blockDim.x * blockIdx.x + threadIdx.x; absIdx < hitsLength; absIdx += gridDim.x * blockDim.x)
	{
		if(d_hits[absIdx] == 0)
		{
			//printf("ZERO KERNEL VAL: %u\n", d_hits[absIdx]);
			*d_zeroes = true;
		}
	}
}


