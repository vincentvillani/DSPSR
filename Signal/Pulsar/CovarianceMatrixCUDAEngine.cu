/*
 * CovarianceMatrixEngineCUDA.C
 *
 *  Created on: 01/09/2014
 *      Author: vincentvillani
 */

#include "dsp/CovarianceMatrixCUDAEngine.h"

//TODO: VINCENT: ADD A HITS CHAN == 1 VARIATION TO STOP NEEDLESS COPYIES


CovarianceMatrixCUDAEngine::CovarianceMatrixCUDAEngine()
{
	h_zeroes = new bool;
	cudaMalloc(&d_zeroes, sizeof(bool));
}


CovarianceMatrixCUDAEngine::~CovarianceMatrixCUDAEngine()
{
	delete h_zeroes;
	cudaFree(d_zeroes);
}



void CovarianceMatrixCUDAEngine::computeCovarianceMatrixCUDAEngine(float* d_result, unsigned int resultElementOffset,
	const float* h_amps, float* d_amps, unsigned int ampsLength,
	const unsigned int* h_hits, unsigned int* d_hits, unsigned int hitsLength,
	unsigned int stokesLength, unsigned int blockDim2D)
{

	cudaMemcpy(d_hits, h_hits, sizeof(unsigned int) * hitsLength, cudaMemcpyHostToDevice);

	//If there are bins with zeroes, discard everything
	if ( hitsContainsZeroes(d_hits, hitsLength) )
		return;


	//printf("RUNNING KERNELS\n");

	int meanBlockDim = blockDim2D * blockDim2D;
	int meanGridDim = ceil((float) ampsLength / meanBlockDim);

	//Copy new amps and hit data to the device
	cudaMemcpy(d_amps, h_amps, sizeof(float) * ampsLength, cudaMemcpyHostToDevice);


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
	outerProductKernel<<< grid, block >>>(d_result + resultElementOffset, d_amps, ampsLength);

	//TODO: DEBUG
	error = cudaDeviceSynchronize();
	if(error != cudaSuccess)
	{
		printf("CUDA ERROR: %s\n", cudaGetErrorString(error));
		exit(2);
	}

}



void CovarianceMatrixCUDAEngine::compute_final_covariance_matrices_device(
		float* d_outerProducts, unsigned int outerProductsLength,
		float* d_runningMeanSum, unsigned int runningMeanSumLength,
		unsigned int unloadCalledCount, unsigned int freqChanNum,
		unsigned int covarianceLength, unsigned int ampsLength)
{
	//check available memory
	size_t freeMemoryBytes;
	size_t totalMemoryBytes;

	cudaMemGetInfo(&freeMemoryBytes, &totalMemoryBytes);

	printf("Free memory: %d\nTotal Memory: %d\n", freeMemoryBytes, totalMemoryBytes);
}




float* CovarianceMatrixCUDAEngine::compute_outer_product_phase_series_device(float* d_runningMeanSum, unsigned int runningMeanSumLength,
			unsigned int unloadCalledCount, unsigned int freqChanNum, unsigned int covarianceLength, unsigned int ampsLength)
{
	/*
	float* d_outerProduct;
	cudaMalloc(&d_outerProduct, sizeof(float) * freqChanNum * covarianceLength);

	//divide the running mean by the number of times unload was called
	unsigned int blockDim = 256;
	unsigned int gridDim = ceil(runningMeanSumLength / blockDim);

	genericDivideKernel<<< gridDim, blockDim >>> (runningMeanSumLength, d_runningMeanSum, unloadCalledCount);

	//Do the outer product

	dim3 outerProductBlockDim = dim3(16, 16);
	dim3 outerProductGridDim = dim3( ceil(runningMeanSumLength  / outerProductBlockDim.x),
									 ceil( (runningMeanSumLength / 2) + 1) /  outerProductBlockDim.y);

	for(int i = 0; i < freqChanNum; ++i)
	{
		outerProductKernel<<< gridDim, blockDim >>> (d_outerProduct + (i * covarianceLength), d_runningMeanSum, runningMeanSumLength);
	}

	*/

	return NULL;
}



bool CovarianceMatrixCUDAEngine::hitsContainsZeroes(unsigned int* d_hits, unsigned int hitLength)
{
	int blockDim = 256;
	int gridDim = ceil((float) hitLength / blockDim);

	//Reset d_zeroes to false
	cudaMemset(d_zeroes, 0, sizeof(bool));

	checkForZeroesKernel<<< gridDim, blockDim >>>(d_hits, hitLength, d_zeroes);
	cudaMemcpy(h_zeroes, d_zeroes, sizeof(bool), cudaMemcpyDeviceToHost);

	return h_zeroes;
}



__global__ void outerProductKernel(float* result, float* vec, int vectorLength)
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
	int index = (row * vectorLength + col) - ((row * (row + 1)) / 2);

	//do the outer product calculation and add it too the correct element
	result[index] += vec[row] * vec[col];
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



__global__ void applyScaleKernel(float* amps, unsigned int ampsLength, double scaleFactor)
{
	int absoluteThreadIdx = blockDim.x * blockIdx.x + threadIdx.x;

	if(absoluteThreadIdx >= ampsLength)
		return;

	amps[absoluteThreadIdx] /= scaleFactor;
}



//----PHASE SERIES COMBINE STUFF----


//Kernel for generically adding things on the GPU
__global__ void genericAddKernel(unsigned int n, float* original, const float* add)
{
	for(int absIdx = blockDim.x * blockIdx.x + threadIdx.x; absIdx < n; absIdx += gridDim.x * blockDim.x)
	{
		original[absIdx] += add[absIdx];
	}
}



//Kernel for generically adding things on the GPU
__global__ void genericAddKernel(unsigned int n, unsigned int* original, const unsigned int* add)
{
	for(int absIdx = blockDim.x * blockIdx.x + threadIdx.x; absIdx < n; absIdx += gridDim.x * blockDim.x)
	{
		original[absIdx] += add[absIdx];
	}
}


__global__ void genericDivideKernel(unsigned int n, float* d_numerators, unsigned int denominator)
{
	for(int absIdx = blockDim.x * blockIdx.x + threadIdx.x; absIdx < n; absIdx += gridDim.x * blockDim.x)
	{
		d_numerators[absIdx] /= denominator;
	}
}



__global__ void checkForZeroesKernel(float* d_hits, unsigned int hitsLength, bool* d_zeroes)
{
	for(int absIdx = blockDim.x * blockIdx.x + threadIdx.x; absIdx < hitsLength; absIdx += gridDim.x * blockDim.x)
	{
		if(d_hits[absIdx] == 0)
			*d_zeroes = true;
	}
}

