/*
 * CovarianceMatrixEngineCuda.C
 *
 *  Created on: 01/09/2014
 *      Author: vincentvillani
 */



#include "dsp/CovarianceMatrixEngineCUDA.h"

void computeCovarianceMatrixCUDA(float* d_resultVector, unsigned int resultByteOffset,
		float* h_amps, float* d_amps, unsigned int ampsLength,
		 float* h_hits, float* d_hits, unsigned int hitsLength,
		 unsigned int stokesLength, unsigned int blockDim2D = 16)
{

	int meanBlockDim = 256;
	int meanGridDim = ceil((float) ampsLength / meanBlockDim);

	//Copy data to device
	cudaMemcpy(d_amps, h_amps, sizeof(float) * ampsLength, cudaMemcpyHostToDevice);
	cudaMemcpy(d_hits, h_hits, sizeof(float) * hitsLength, cudaMemcpyHostToDevice);


	meanStokesKernel<<< meanGridDim, meanBlockDim >>>(d_amps, ampsLength, d_hits);


	//Compute the needed block and grid dimensions
	int blockDimX = blockDim2D;
	int blockDimY = blockDim2D;
	int gridDimX = ceil((float) ampsLength / blockDimX);
	int gridDimY = ceil((float) ((ampsLength / 2) + 1) / blockDimY);

	dim3 grid = dim3(gridDimX, gridDimY);
	dim3 block = dim3(blockDimX, blockDimY);

	//Call the kernel
	//amps now contains the mean stokes elements
	outerProductKernel<<< grid, block >>>(d_resultVector, d_amps, ampsLength);

}
