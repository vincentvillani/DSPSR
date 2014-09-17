/*
 * CovarianceMatrixEngineCUDA.C
 *
 *  Created on: 01/09/2014
 *      Author: vincentvillani
 */

#include "dsp/CovarianceMatrixCUDAEngine.h"

//TODO: VINCENT: ADD A HITS CHAN == 1 VARIATION TO STOP NEEDLESS COPYIES

//#if HAVE_CUDA
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
//#endif


dsp::CovarianceMatrixCUDAEngine::CovarianceMatrixCUDAEngine()
{
	cudaMalloc(&d_zeroes, sizeof(bool));
	h_zeroes = false;
}


dsp::CovarianceMatrixCUDAEngine::~CovarianceMatrixCUDAEngine()
{
	cudaFree(d_zeroes);
}


//SINGLE HIT DIM
void dsp::CovarianceMatrixCUDAEngine::computeCovarianceMatricesCUDA(const PhaseSeries* ps, CovarianceMatrixResult* cmr)
{
	unsigned int chanNum = ps->get_nchan();
	unsigned int hitsLength = cmr->getHitsLength();

	unsigned int* d_hits = cmr->getHits();
	const unsigned int* h_hits = getHitsPtr(ps, cmr, 0);
	gpuErrchk( cudaMemcpy(d_hits, h_hits, sizeof(unsigned int) * hitsLength, cudaMemcpyHostToDevice) );

	//If there are bins with zeroes, discard everything
	if ( hitsContainsZeroes(d_hits, hitsLength) )
	{
		printf("There are bins with zeroes, returning...\n");
		return;
	}



	//For each channel, compute the covariance matrix
	for(unsigned int i = 0; i < chanNum; ++i)
	{

		const float* h_amps = ps->get_datptr(i, 0);

		computeCovarianceMatrix(cmr->getCovarianceMatrix(i),
				h_amps, cmr->getAmps(), cmr->getAmpsLength(),
				cmr->getHits(), cmr->getHitsLength(),
				cmr->getRunningMeanSum(i),
				cmr->getStokesLength());


		//TODO: VINCENT: DEBUG
		if(i == 0)
		{
			float val;
			cudaMemcpy(&val, cmr->getCovarianceMatrix(0), sizeof(float), cudaMemcpyDeviceToHost);
			printf("Value: %f\n", val);
		}

	}

	cmr->getPhaseSeries()->combine(ps);

}



void dsp::CovarianceMatrixCUDAEngine::computeCovarianceMatrix(float* d_result,
	const float* h_amps, float* d_amps, unsigned int ampsLength,
	unsigned int* d_hits, unsigned int hitsLength,
	float* d_runningMean,
	unsigned int stokesLength, unsigned int blockDim2D)
{

	int meanBlockDim = blockDim2D * blockDim2D;
	int meanGridDim = ceil((float) ampsLength / meanBlockDim);

	//Copy new amps and hit data to the device
	gpuErrchk(cudaMemcpy(d_amps, h_amps, sizeof(float) * ampsLength, cudaMemcpyHostToDevice));


	//h_hits values should be copied over to d_hits before this function is called
	//printf("Launching Mean Kernel with gridDim: %d, blockDim: %d\n", meanGridDim, meanBlockDim);
	meanStokesKernel <<< meanGridDim, meanBlockDim >>> (d_amps, ampsLength, d_hits, stokesLength);

	//TODO: DEBUG
	cudaError_t error = cudaPeekAtLastError();
	if(error != cudaSuccess)
	{
		printf("CUDA ERROR: %s\n", cudaGetErrorString(error));
		exit(1);
	}


	genericAddKernel <<< meanGridDim, meanBlockDim >>> (ampsLength, d_runningMean, d_amps);


	//TODO: DEBUG
	error = cudaPeekAtLastError();
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

	dim3 block = dim3(blockDimX, blockDimY);
	dim3 grid = dim3(gridDimX, gridDimY);

	//Call the kernel
	//Compute covariance matrix
	//printf("Launching outerProduct Kernel with gridDim: (%d, %d), blockDim: (%d, %d)\n\n",
			//grid.x, grid.y, block.x, block.y);
	outerProductKernel <<< grid, block >>>(d_result, d_amps, ampsLength);

	//TODO: DEBUG
	cudaError_t error2 = cudaPeekAtLastError();
	if(error2 != cudaSuccess)
	{
		printf("CUDA ERROR: %s\n", cudaGetErrorString(error2));
		exit(2);
	}

}






float* dsp::CovarianceMatrixCUDAEngine::compute_final_covariance_matrices_device(CovarianceMatrixResult* cmr)
{
	//Compute the phase series outer products
	float* d_phaseSeriesOuterProduct = compute_outer_product_phase_series_device(cmr);

	unsigned int totalElementLength = cmr->getCovarianceMatrixLength() * cmr->getNumberOfFreqChans();
	unsigned int blockDim = 256;
	unsigned int gridDim = min (ceil ( totalElementLength / blockDim), (double)((1 << 16) - 1)); //number of elements / blockdim

	//Divide all x^2 terms by unload call count
	printf("Launching generic divide kernel with gridDim: %u, blockDim: %u\n", gridDim, blockDim);
	genericDivideKernel <<< gridDim, blockDim >>> (totalElementLength, cmr->getCovarianceMatrix(0), cmr->getUnloadCallCount());

	//TODO: VINCENT: DEBUG
	cudaError_t error = cudaPeekAtLastError();
	if(error != cudaSuccess)
	{
		printf("CUDA ERROR: %s\n", cudaGetErrorString(error));
		exit(2);
	}


	genericSubtractionKernel <<< gridDim, blockDim >>> (totalElementLength, cmr->getCovarianceMatrix(0), d_phaseSeriesOuterProduct);

	//TODO: VINCENT: DEBUG
	cudaError_t error2 = cudaPeekAtLastError();
	if(error2 != cudaSuccess)
	{
		printf("CUDA ERROR2: %s\n", cudaGetErrorString(error2));
		exit(2);
	}

	cudaFree(d_phaseSeriesOuterProduct);

	float* h_outerProduct = new float[totalElementLength];
	cudaMemcpy(h_outerProduct, cmr->getCovarianceMatrix(0), sizeof(float) * totalElementLength, cudaMemcpyDeviceToHost);

	return h_outerProduct;
}




float* dsp::CovarianceMatrixCUDAEngine::compute_outer_product_phase_series_device(CovarianceMatrixResult* cmr)
{

	unsigned int totalCovarianceLength = cmr->getCovarianceMatrixLength() * cmr-> getNumberOfFreqChans();

	float* d_runningMeanSum = cmr->getRunningMeanSum(0);
	unsigned int runningMeanSumLength = cmr->getRunningMeanSumLength();

	float* d_outerProduct;
	cudaMalloc(&d_outerProduct, sizeof(float) * totalCovarianceLength);
	cudaMemset(d_outerProduct, 0, sizeof(float) * totalCovarianceLength);

	//divide the running mean by the number of times unload was called
	unsigned int blockDim = 256;
	unsigned int gridDim = ceil(runningMeanSumLength / blockDim);

	printf("Starting generic divide kernel - GridDim: %u, BlockDim: %u\n", gridDim, blockDim);
	genericDivideKernel<<< gridDim, blockDim >>> (runningMeanSumLength, d_runningMeanSum, cmr->getUnloadCallCount());

	//TODO: VINCENT: DEBUG
	cudaError_t error = cudaPeekAtLastError();
	if(error != cudaSuccess)
	{
		printf("CUDA ERROR: %s\n", cudaGetErrorString(error));
		exit(2);
	}

	//Do the outer product

	dim3 outerProductBlockDim = dim3(16, 16);
	dim3 outerProductGridDim = dim3( ceil(runningMeanSumLength  / outerProductBlockDim.x),
									 ceil( (runningMeanSumLength / 2) + 1) /  outerProductBlockDim.y);

	unsigned int oneFreqRunningMeanLength = cmr->getBinNum() * cmr->getStokesLength();

	for(unsigned int i = 0; i < cmr->getNumberOfFreqChans(); ++i)
	{
		printf("Starting outer product kernel - GridDim: (%u, %u) BlockDim: (%u, %u)\n", outerProductGridDim.x, outerProductGridDim.y,
				outerProductBlockDim.x, outerProductBlockDim.y);

		outerProductKernel<<< outerProductGridDim, outerProductBlockDim >>> (d_outerProduct + (i * cmr->getCovarianceMatrixLength()),
				d_runningMeanSum + (i * oneFreqRunningMeanLength), oneFreqRunningMeanLength);

		//TODO: VINCENT: DEBUG
		cudaError_t error2 = cudaPeekAtLastError();
		if(error2 != cudaSuccess)
		{
			printf("CUDA ERROR: %s\n", cudaGetErrorString(error2));
			exit(2);
		}
	}


	float* h_outerProduct = new float[cmr->getCovarianceMatrixLength() * cmr->getNumberOfFreqChans()];
	cudaMemcpy(h_outerProduct, d_outerProduct, sizeof(float) * cmr->getCovarianceMatrixLength() * cmr->getNumberOfFreqChans(), cudaMemcpyDeviceToHost);


	//**** DEBUG ****** TODO:VINCENT: DEBUG
	std::stringstream ss;

	//Write out data to a file
	for(int j = 0; j < cmr->getNumberOfFreqChans(); ++j)
	{
		//write it out to a file
		ss << "xMeanGPU" << j << ".txt";
		outputUpperTriangularMatrix(h_outerProduct + (j * cmr->getCovarianceMatrixLength()),
				cmr->getBinNum() * cmr->getStokesLength(), ss.str());
		ss.str("");
	}


	return d_outerProduct;
}



bool dsp::CovarianceMatrixCUDAEngine::hitsContainsZeroes(unsigned int* d_hits, unsigned int hitLength)
{
	int blockDim = 256;
	int gridDim = ceil((float) hitLength / blockDim);

	//Reset d_zeroes to false
	cudaMemset(d_zeroes, 0, sizeof(bool));


	checkForZeroesKernel<<< gridDim, blockDim >>> (d_hits, hitLength, d_zeroes);

	//TODO: VINCENT: DEBUG
	cudaError_t error2 = cudaPeekAtLastError();
	if(error2 != cudaSuccess)
	{
		printf("CUDA ERROR: %s\n", cudaGetErrorString(error2));
		exit(2);
	}


	gpuErrchk(cudaMemcpy(&h_zeroes, d_zeroes, sizeof(bool), cudaMemcpyDeviceToHost));

	return h_zeroes;
}



const unsigned int* dsp::CovarianceMatrixCUDAEngine::getHitsPtr(const PhaseSeries* phaseSeriesData, CovarianceMatrixResult* covarianceMatrixResult, int freqChan)
{
	//return the only channel
	if(covarianceMatrixResult->getNumberOfHitChans() == 1)
	{
		//printf("ONE HIT CHAN!\n");

		return phaseSeriesData->get_hits(0);
	}
	else
		return phaseSeriesData->get_hits(freqChan); //Return the hits pointer using the freq channel
}




void dsp::CovarianceMatrixCUDAEngine::outputUpperTriangularMatrix(float* result, unsigned int rowLength, std::string filename)
{

	FILE* file = fopen(filename.c_str(), "w");

	int numZeros = 0;
	int iterator = 0;

	//for every row
	for(int i = 0; i < rowLength; ++i)
	{
		//print preceding zeros
		for(int j = 0; j < numZeros; ++j)
		{
			fprintf(file, "0 ");
		}

		//print array values
		for(int k = 0; k < rowLength - numZeros; ++k)
		{
			fprintf(file, "%f ", result[iterator]);
			++iterator;
		}

		fprintf(file, "\n");
		numZeros++;
	}

	fclose(file);

}




__global__ void outerProductKernel(float* result, float* vec, unsigned int vectorLength)
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


//(d_amps, ampsLength, d_hits, stokesLength)
__global__ void meanStokesKernel(float* d_amps, unsigned int ampsLength, unsigned int* d_hits, unsigned int stokesLength)
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
__global__ void genericAddKernel(unsigned int n, unsigned int* original, const unsigned int* add)
{
	for(unsigned int absIdx = blockDim.x * blockIdx.x + threadIdx.x; absIdx < n; absIdx += gridDim.x * blockDim.x)
	{
		original[absIdx] += add[absIdx];
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



__global__ void checkForZeroesKernel(unsigned int* d_hits, unsigned int hitsLength, bool* d_zeroes)
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

