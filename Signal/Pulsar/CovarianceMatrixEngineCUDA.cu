/*
 * CovarianceMatrixEngineCUDA.C
 *
 *  Created on: 01/09/2014
 *      Author: vincentvillani
 */



#include "dsp/CovarianceMatrixEngineCUDA.h"

void computeCovarianceMatrixCUDAEngine(float* d_resultVector, unsigned int resultElementOffset,
		const float* h_amps, float* d_amps, unsigned int ampsLength,
		 const unsigned int* h_hits, unsigned int* d_hits, unsigned int hitsLength,
		 unsigned int stokesLength, double scaleFactor, unsigned int blockDim2D)
{

	printf("RUNNING KERNELS\n");

	int meanBlockDim = 256;
	int meanGridDim = ceil((float) ampsLength / meanBlockDim);

	//Copy data to device
	cudaMemcpy(d_amps, h_amps, sizeof(float) * ampsLength, cudaMemcpyHostToDevice);
	cudaMemcpy(d_hits, h_hits, sizeof(unsigned int) * hitsLength, cudaMemcpyHostToDevice);

	//unsigned int hitZero;
	//cudaMemcpy(&hitZero, d_hits, sizeof(unsigned int), cudaMemcpyDeviceToHost);

	//printf("Hit Zero: %d\n", hitZero);

	//printf("Launching scale Kernel with gridDim: %d, blockDim: %d\n", meanGridDim, meanBlockDim);

	applyScale <<< meanGridDim, meanBlockDim >>> (d_amps, ampsLength, scaleFactor);


	//printf("Launching Mean Kernel with gridDim: %d, blockDim: %d\n", meanGridDim, meanBlockDim);

	//float ampZero;

	//cudaMemcpy(&ampZero, d_amps, sizeof(float), cudaMemcpyDeviceToHost);

	//printf("After: amp zero: %f\n", ampZero);
	//printf("After: hit zero: %d\n", h_hits[0]);


	meanStokesKernel<<< meanGridDim, meanBlockDim >>>(d_amps, ampsLength, d_hits, stokesLength);

	//TODO: DEBUG
	cudaError_t error = cudaDeviceSynchronize();
	if(error != cudaSuccess)
	{
		printf("CUDA ERROR: %s\n", cudaGetErrorString(error));
	}

	//Compute the needed block and grid dimensions
	int blockDimX = blockDim2D;
	int blockDimY = blockDim2D;
	int gridDimX = ceil((float) ampsLength / blockDimX);
	int gridDimY = ceil((float) ((ampsLength / 2) + 1) / blockDimY);

	dim3 grid = dim3(gridDimX, gridDimY);
	dim3 block = dim3(blockDimX, blockDimY);

	printf("Launching outerProduct Kernel with gridDim: (%d, %d), blockDim: (%d, %d)\n",
			grid.x, grid.y, block.x, block.y);

	//Call the kernel
	//Compute covariance matrix
	outerProductKernel<<< grid, block >>>(d_resultVector + resultByteOffset, d_amps, ampsLength);

	//TODO: DEBUG
	error = cudaDeviceSynchronize();
	if(error != cudaSuccess)
	{
		printf("CUDA ERROR: %s\n", cudaGetErrorString(error));
	}

}
