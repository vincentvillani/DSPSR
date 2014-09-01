/*
 * CovarianceMatrixEngineCuda.C
 *
 *  Created on: 01/09/2014
 *      Author: vincentvillani
 */



#include "CovarianceMatrixEngineCuda.h"

void computeCovarianceMatrixCUDA(float* d_resultVector, float* d_vector, unsigned int vectorLength,
		unsigned int blockDim2D)
{
	//Compute the needed block and grid dimensions
	int blockDimX = blockDim;
	int blockDimY = blockDim;
	int gridDimX = ceil((float) vecNCol / blockDimX);
	int gridDimY = ceil((float) ((vecNCol / 2) + 1) / blockDimY);

	dim3 grid = dim3(gridDimX, gridDimY);
	dim3 block = dim3(blockDimX, blockDimY);

	//Call the kernel
	outerProductKernel<<<grid, block>>>(d_resultVector, d_vector, vectorLength);
}
