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

	if(col == 0 && row == 0)
	{
		printf("CUDA RESULT BEFORE: %f\n", resultMatrix[index]);
	}

	//do the outer product calculation and add it too the correct element
	resultMatrix[index] += vec[row] * vec[col];

	if(col == 0 && row == 0)
	{
		printf("CUDA RESULT AFTER: %f\n", resultMatrix[index]);
	}
}



__global__ void meanStokesKernel(float* d_amps, unsigned int ampsLength, unsigned int* d_hits, unsigned int stokesLength)
{
	int absoluteThreadIdx = blockDim.x * blockIdx.x + threadIdx.x;

	if(absoluteThreadIdx >= ampsLength)
		return;

	if(absoluteThreadIdx == 0)
		printf("hitVal BEFORE: %d\n", d_hits[0]);

	float hitVal = d_hits[ absoluteThreadIdx / stokesLength ];

	if(absoluteThreadIdx == 0)
		printf("hitVal AFTER: %d\n", hitVal);

	//can't divide by zero so just return
	if(hitVal == 0.0f)
		return;

	d_amps[absoluteThreadIdx] = d_amps[absoluteThreadIdx] / hitVal;

	if(absoluteThreadIdx == 0)
		printf("AMPS MEAN[0]: %f\n", d_amps[0]);

}



__global__ void applyScale(float* amps, unsigned int ampsLength, double scaleFactor)
{
	int absoluteThreadIdx = blockDim.x * blockIdx.x + threadIdx.x;

	if(absoluteThreadIdx >= ampsLength)
		return;

	amps[absoluteThreadIdx] = amps[absoluteThreadIdx] / scaleFactor;
}

