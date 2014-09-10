/*
 * CovarianceMatrixCuda.cu
 *
 *  Created on: 01/09/2014
 *      Author: vincentvillani
 */

#include "dsp/CovarianceMatrixCUDA.h"

__global__ void outerProductKernel(float* resultMatrix, float* vec, int vectorLength)
{
	int col = (blockIdx.x * blockDim.x) + threadIdx.x; //column
	int row = (blockIdx.y * blockDim.y) + threadIdx.y; //row

	//check bounds
	if(row >= vectorLength || col >= vectorLength)
		return;

	//transpose
	if(row > col)
	{
		row = vectorLength - row;
		col = row + col;
	}

	//compute the index
	int index = (row * vectorLength + col) - (row * (row + 1)) / 2;

	//do the outer product calculation and add it too the correct element
	resultMatrix[index] += vec[row] * vec[col];
}



__global__ void meanStokesKernel(float* d_amps, unsigned int ampsLength, unsigned int* d_hits, unsigned int stokesLength)
{
	int absoluteThreadIdx = blockDim.x * blockIdx.x + threadIdx.x;

	if(absoluteThreadIdx >= ampsLength)
		return;

	unsigned int hitVal = d_hits[ absoluteThreadIdx / stokesLength ];

	//can't divide by zero so just return
	if(hitVal == 0)
	{
		d_amps[absoluteThreadIdx] = 0;
		return;
	}

	d_amps[absoluteThreadIdx] = d_amps[absoluteThreadIdx] / (float)hitVal;

}



__global__ void applyScale(float* amps, unsigned int ampsLength, double scaleFactor)
{
	int absoluteThreadIdx = blockDim.x * blockIdx.x + threadIdx.x;

	if(absoluteThreadIdx >= ampsLength)
		return;

	amps[absoluteThreadIdx] = amps[absoluteThreadIdx] / scaleFactor;
}



//----PHASE SERIES COMBINE STUFF----


//Kernel for generically adding things on the GPU
__global__ void genericAdd(uint64_t n, float* original, const float* add)
{
	for(int absIdx = blockDim.x * blockIdx.x + threadIdx.x; absIdx < n; absIdx += gridDim.x * blockDim.x)
	{
		originalTS[absIdx] += addTS[absIdx];
	}
}


//Kernel for generically adding things on the GPU
__global__ void genericAdd(unsigned int n, unsigned int* original, const unsigned int* add)
{
	for(int absIdx = blockDim.x * blockIdx.x + threadIdx.x; absIdx < n; absIdx += gridDim.x * blockDim.x)
	{
		originalTS[absIdx] += addTS[absIdx];
	}
}



