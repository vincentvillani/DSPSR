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


	computeCovarianceMatrix(cmr, ps);


	//TODO: VINCENT: DEBUG
	float val;
	cudaMemcpy(&val, cmr->getCovarianceMatrix(0), sizeof(float), cudaMemcpyDeviceToHost);
	printf("Value: %f\n", val);

	cmr->getPhaseSeries()->combine(ps);

}



void dsp::CovarianceMatrixCUDAEngine::computeCovarianceMatrix(CovarianceMatrixResult* cmr, const PhaseSeries* ps)
{
	unsigned int ampsLength = cmr->getAmpsLength();
	unsigned int covMatrixLength = cmr->getCovarianceMatrixLength();
	unsigned int stokesLength = cmr->getStokesLength();


	float* d_amps = cmr->getAmps();
	unsigned int* d_hits = cmr->getHits();
	float* d_runningMean;
	float* d_result;


	unsigned int meanBlockDim = 256;
	unsigned int meanGridDim =  ceil( ampsLength / meanBlockDim);
	unsigned int outerProductBlockSize = 256;
	unsigned int outerProductGridDim = min( (int)ceil( (int)((ampsLength * (ampsLength + 1)) / 2) / outerProductBlockSize), 65535);


	//compute the covariance matrix for each freq chan
	for(unsigned int i = 0; i < cmr->getNumberOfFreqChans(); ++i)
	{
		//first normalise/compute the mean of the amps by dividing it by the hits
		const float* h_amps = ps->get_datptr(i, 0);
		gpuErrchk(cudaMemcpy(d_amps, h_amps + (i * ampsLength), sizeof(float) * ampsLength, cudaMemcpyHostToDevice));

		//h_hits values should be copied over to d_hits before this function is called <------------ IS THIS TRUE?????????!!!!!?!?!?!!??
		printf("Launching Mean Kernel with gridDim: %d, blockDim: %d\n", meanGridDim, meanBlockDim);
		meanStokesKernel <<< meanGridDim, meanBlockDim >>> (d_amps, ampsLength, d_hits, stokesLength);

		//TODO: DEBUG
		cudaError_t error = cudaPeekAtLastError();
		if(error != cudaSuccess)
		{
			printf("CUDA ERROR: %s\n", cudaGetErrorString(error));
			exit(1);
		}


		//Add the normalised amps to the running mean
		d_runningMean = cmr->getRunningMeanSum(i);
		genericAddKernel <<< meanGridDim, meanBlockDim >>> (ampsLength, d_runningMean, d_amps);

		//TODO: DEBUG
		error = cudaPeekAtLastError();
		if(error != cudaSuccess)
		{
			printf("CUDA ERROR: %s\n", cudaGetErrorString(error));
			exit(1);
		}



		//Compute the outer product
		printf("Launching outerProduct Kernel with gridDim: %u, blockDim: %u\n\n",
				outerProductGridDim, outerProductBlockSize);
		d_result = cmr->getCovarianceMatrix(i);
		outerProductKernelNew <<<outerProductGridDim, outerProductBlockSize>>>
				(d_result, covMatrixLength, d_amps, ampsLength);

	}

	//combine phase series
	cmr->getPhaseSeries()->combine(ps);
}



float* dsp::CovarianceMatrixCUDAEngine::compute_final_covariance_matrices_device_DEBUG(CovarianceMatrixResult* cmr)
{
	printf("BEFORE PSOP\n");

	float* phaseSeriesOuterProduct = compute_outer_product_phase_series_device_DEBUG(cmr);

	printf("AFTER PSOP\n");

	unsigned int freqChanNum = cmr->getNumberOfFreqChans();
	unsigned int covarianceMatrixLength = cmr->getCovarianceMatrixLength();
	unsigned int unloadCalledNum = cmr->getUnloadCallCount();

	float* covarianceMatrix = new float[covarianceMatrixLength];

	for(int i = 0; i < freqChanNum; ++i)
	{
		float* d_covarianceMatrix = cmr->getCovarianceMatrix(i);
		gpuErrchk(cudaMemcpy(covarianceMatrix, d_covarianceMatrix, sizeof(float) * covarianceMatrixLength, cudaMemcpyDeviceToHost));

		for(int j = 0; j < covarianceMatrixLength; ++j)
		{

			covarianceMatrix[j] /= unloadCalledNum;
			covarianceMatrix[j] -= phaseSeriesOuterProduct[(i * covarianceMatrixLength) + j];
		}

		//copy results back
		gpuErrchk(cudaMemcpy(d_covarianceMatrix, covarianceMatrix, sizeof(float) * covarianceMatrixLength, cudaMemcpyHostToDevice));
	}

	printf("HERE\n");



	float* finalCov = new float[covarianceMatrixLength * freqChanNum];
	gpuErrchk(cudaMemcpy(finalCov, cmr->getCovarianceMatrix(0), sizeof(float) * covarianceMatrixLength * freqChanNum, cudaMemcpyDeviceToHost));

	printf("DONE\n");

	delete[] covarianceMatrix;
	delete[] phaseSeriesOuterProduct;
	return finalCov;

}



float* dsp::CovarianceMatrixCUDAEngine::compute_final_covariance_matrices_device(CovarianceMatrixResult* cmr)
{

	//Compute the phase series outer products
	float* d_phaseSeriesOuterProduct = compute_outer_product_phase_series_device(cmr);

	unsigned int totalElementLength = cmr->getCovarianceMatrixLength() * cmr->getNumberOfFreqChans();
	unsigned int blockDim = 256;
	unsigned int gridDim = min (ceil ( totalElementLength / blockDim), (double) 65535); //number of elements / blockdim

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

	unsigned int ampsLength = cmr->getAmpsLength();
	unsigned int outerProductBlockDim = 256;
	unsigned int outerProductGridDim = min( (int)ceil( (int)((ampsLength * (ampsLength + 1)) / 2) / outerProductBlockDim), 65535);

	//unsigned int oneFreqRunningMeanLength = cmr->getBinNum() * cmr->getStokesLength();

	for(unsigned int i = 0; i < cmr->getNumberOfFreqChans(); ++i)
	{
		printf("Starting outer product kernel - GridDim: %u, BlockDim: %u\n", outerProductGridDim, outerProductBlockDim);
		outerProductKernelNew <<< outerProductGridDim, outerProductBlockDim >>>
				(d_outerProduct + (i * cmr->getCovarianceMatrixLength()), cmr->getCovarianceMatrixLength(),
						d_runningMeanSum + (i * ampsLength), ampsLength);

		//TODO: VINCENT: DEBUG
		cudaError_t error2 = cudaPeekAtLastError();
		if(error2 != cudaSuccess)
		{
			printf("CUDA ERROR: %s\n", cudaGetErrorString(error2));
			exit(2);
		}
	}


	return d_outerProduct;
}




float* dsp::CovarianceMatrixCUDAEngine::compute_outer_product_phase_series_device_DEBUG(CovarianceMatrixResult* cmr)
{
	unsigned int unloadCallCount = cmr->getUnloadCallCount();
	unsigned int freqChanNum = cmr->getNumberOfFreqChans();
	unsigned int covarianceLength = cmr->getCovarianceMatrixLength();
	unsigned int ampsLength = cmr->getBinNum() * cmr->getStokesLength();

	float* outerProduct = new float [freqChanNum * covarianceLength];
	float* runningMeanSum = new float[ampsLength];


	//For each freq channel
	for(unsigned int channel = 0; channel < freqChanNum; ++channel)
	{

		float* d_runningMeanSum = cmr->getRunningMeanSum(channel);
		cudaMemcpy(runningMeanSum, d_runningMeanSum, sizeof(float) * ampsLength, cudaMemcpyDeviceToHost);


		//divide running mean sum by number of times called
		for(unsigned int i = 0; i < ampsLength; ++i)
		{
			runningMeanSum[i] /= unloadCallCount;
		}


		//Do the outer product
		for(unsigned int row = 0; row < ampsLength; ++row)
		{
			for(unsigned int col = row; col < ampsLength; ++col)
			{
				outerProduct[ (channel * covarianceLength) +  ((row * ampsLength + col) - ( (row * (row + 1)) / 2) ) ] =
						runningMeanSum[row] * runningMeanSum[col];
			}
		}

	}


	return outerProduct;
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
		return phaseSeriesData->get_hits(0);
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



__global__ void outerProductKernelNew(float* result, unsigned int resultLength, float* vec, unsigned int vecLength)
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

