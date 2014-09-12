/*
 * CovarianceMatrixEngineCUDA.C
 *
 *  Created on: 01/09/2014
 *      Author: vincentvillani
 */

#include "dsp/CovarianceMatrixCUDAEngine.h"



__host__ void CovarianceMatrixCUDAEngine::computeCovarianceMatrixCUDAEngine(float* d_resultVector, unsigned int resultElementOffset,
		const float* h_amps, float* d_amps, unsigned int ampsLength,
		 const unsigned int* h_hits, unsigned int* d_hits, unsigned int hitsLength,
		 unsigned int stokesLength, double scaleFactor, unsigned int blockDim2D)
{

	//printf("RUNNING KERNELS\n");

	int meanBlockDim = 256;
	int meanGridDim = ceil((float) ampsLength / meanBlockDim);

	//Copy data to device
	cudaMemcpy(d_amps, h_amps, sizeof(float) * ampsLength, cudaMemcpyHostToDevice);
	cudaMemcpy(d_hits, h_hits, sizeof(unsigned int) * hitsLength, cudaMemcpyHostToDevice);

	printf("Launching scale Kernel with gridDim: %d, blockDim: %d\n", meanGridDim, meanBlockDim);
	//applyScale <<< meanGridDim, meanBlockDim >>> (d_amps, ampsLength, scaleFactor);


	printf("Launching Mean Kernel with gridDim: %d, blockDim: %d\n", meanGridDim, meanBlockDim);
	meanStokesKernel<<< meanGridDim, meanBlockDim >>>(d_amps, ampsLength, d_hits, stokesLength);

	//TODO: DEBUG
	cudaError_t error = cudaDeviceSynchronize();
	if(error != cudaSuccess)
	{
		printf("CUDA ERROR: %s\n", cudaGetErrorString(error));
		exit(1);
	}

	//Compute the needed block and grid dimensions
	int blockDimX = blockDim2D;
	int blockDimY = blockDim2D;
	int gridDimX = ceil((float) ampsLength / blockDimX);
	int gridDimY = ceil((float) ((ampsLength / 2) + 1) / blockDimY);

	dim3 grid = dim3(gridDimX, gridDimY);
	dim3 block = dim3(blockDimX, blockDimY);

	//Call the kernel
	//Compute covariance matrix
	printf("Launching outerProduct Kernel with gridDim: (%d, %d), blockDim: (%d, %d)\n\n",
			grid.x, grid.y, block.x, block.y);
	outerProductKernel<<< grid, block >>>(d_resultVector + resultElementOffset, d_amps, ampsLength);

	//TODO: DEBUG
	error = cudaDeviceSynchronize();
	if(error != cudaSuccess)
	{
		printf("CUDA ERROR: %s\n", cudaGetErrorString(error));
		exit(1);
	}

}



__global__ void CovarianceMatrixCUDAEngine::outerProductKernel(float* resultMatrix, float* vec, int vectorLength)
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



__global__ void CovarianceMatrixCUDAEngine::meanStokesKernel(float* d_amps, unsigned int ampsLength, unsigned int* d_hits, unsigned int stokesLength)
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



__global__ void CovarianceMatrixCUDAEngine::applyScale(float* amps, unsigned int ampsLength, double scaleFactor)
{
	int absoluteThreadIdx = blockDim.x * blockIdx.x + threadIdx.x;

	if(absoluteThreadIdx >= ampsLength)
		return;

	amps[absoluteThreadIdx] = amps[absoluteThreadIdx] / scaleFactor;
}



//----PHASE SERIES COMBINE STUFF----


//Kernel for generically adding things on the GPU
__global__ void CovarianceMatrixCUDAEngine::genericAdd(unsigned int n, float* original, const float* add)
{
	for(int absIdx = blockDim.x * blockIdx.x + threadIdx.x; absIdx < n; absIdx += gridDim.x * blockDim.x)
	{
		originalTS[absIdx] += addTS[absIdx];
	}
}


//Kernel for generically adding things on the GPU
__global__ void CovarianceMatrixCUDAEngine::genericAdd(unsigned int n, unsigned int* original, const unsigned int* add)
{
	for(int absIdx = blockDim.x * blockIdx.x + threadIdx.x; absIdx < n; absIdx += gridDim.x * blockDim.x)
	{
		originalTS[absIdx] += addTS[absIdx];
	}
}

