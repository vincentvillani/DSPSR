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



__global__ void meanStokesKernel(float* amps, unsigned int ampsLength, float* hits, unsigned int stokesLength)
{
	int absoluteThreadIdx = blockDim.x * blockIdx.x + threadIdx.x;

	if(absoluteThreadIdx >= ampsLength)
		return;

	float hitVal = hits[ absoluteThreadIdx / stokesLength ];

	//can't divide by zero so just return
	if(hitVal == 0.0f)
		return;

	amps[absoluteThreadIdx] = 1;//amps[absoluteThreadIdx] / hitVal;

}

