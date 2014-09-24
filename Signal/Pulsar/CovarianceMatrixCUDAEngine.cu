/*
 * CovarianceMatrixEngineCUDA.C
 *
 *  Created on: 01/09/2014
 *      Author: vincentvillani
 */

#include "dsp/CovarianceMatrixCUDAEngine.h"


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



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
	//const unsigned int* d_hits = cmr->getHits();
	unsigned int hitChanNum = cmr->getNumberOfHitChans();


	PhaseSeries clonedPhaseSeries;
	clonedPhaseSeries.set_hits_memory(new CUDA::DeviceMemory());
	clonedPhaseSeries = *ps;

	if( ps->get_memory()->on_host() )
		printf("CLONED PHASE SERIES ON HOST!\n");
	else
		printf("CLONED PHASE SERIES NOT ON HOST!\n");




	computeCovarianceMatrix(cmr, &clonedPhaseSeries);


	for(unsigned int chan = 0; chan < hitChanNum; ++chan)
	{
		unsigned int* d_hits = getHitsPtr(&clonedPhaseSeries, cmr, chan); // TODO: VINCENT: Are hit chans guaranteed to be next to each other? if so I can just copy all at once
		//gpuErrchk( cudaMemcpy(d_hits + (chan * hitsLength), h_hits, sizeof(unsigned int) * hitsLength, cudaMemcpyHostToDevice) ); 	//Copy the hits data over to the device

		//If there are bins with zeroes, discard everything
		if ( hitsContainsZeroes(d_hits + (chan * hitsLength), hitsLength) )
		{
			printf("There are bins with zeroes, returning...\n");
			return;
		}

	}



	//TODO: VINCENT: DEBUG
	float val;
	cudaMemcpy(&val, cmr->getCovarianceMatrix(0), sizeof(float), cudaMemcpyDeviceToHost);
	printf("Value: %f\n", val);

	cmr->getPhaseSeries()->combine(ps);

}



void dsp::CovarianceMatrixCUDAEngine::computeCovarianceMatrix(CovarianceMatrixResult* cmr, PhaseSeries* ps)
{
	unsigned int ampsLength = cmr->getAmpsLength();
	unsigned int covMatrixLength = cmr->getCovarianceMatrixLength();
	unsigned int stokesLength = cmr->getStokesLength();


	unsigned int* d_hits = cmr->getHits();
	float* d_runningMean;
	float* d_result;

	unsigned int meanBlockDim = 256;
	unsigned int meanGridDim =  ceil( ampsLength / meanBlockDim);
	unsigned int outerProductBlockSize = 256;
	unsigned int outerProductGridDim = min( (int)ceil( (int)((ampsLength * (ampsLength + 1)) / 2) / outerProductBlockSize), 65535);


	//combine phase series
	cmr->getPhaseSeries()->combine(ps);

	//compute the covariance matrix for each freq chan
	for(unsigned int i = 0; i < cmr->getNumberOfFreqChans(); ++i)
	{

		if(cmr->getNumberOfHitChans() != 1)
		{
			d_hits += cmr->getHitsLength(); //move d_hits pointer by the appropriate amount to get the next channels data
		}

		//first normalise/compute the mean of the amps by dividing it by the hits
		float* d_amps = ps->get_datptr(i, 0);
		//gpuErrchk(cudaMemcpy(d_amps, h_amps + (i * ampsLength), sizeof(float) * ampsLength, cudaMemcpyHostToDevice));

		//h_hits values should be copied over to d_hits before this function is called
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
		outerProductKernel <<<outerProductGridDim, outerProductBlockSize>>>
				(d_result, covMatrixLength, d_amps, ampsLength);

	}


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

	for(unsigned int i = 0; i < cmr->getNumberOfFreqChans(); ++i)
	{
		printf("Starting outer product kernel - GridDim: %u, BlockDim: %u\n", outerProductGridDim, outerProductBlockDim);
		outerProductKernel <<< outerProductGridDim, outerProductBlockDim >>>
				(cmr->getCovarianceMatrix(i), cmr->getCovarianceMatrixLength(),
						cmr->getRunningMeanSum(i), ampsLength);

		//TODO: VINCENT: DEBUG
		cudaError_t error2 = cudaPeekAtLastError();
		if(error2 != cudaSuccess)
		{
			printf("CUDA ERROR: %s\n", cudaGetErrorString(error2));
			exit(2);
		}
	}


	return cmr->getCovarianceMatrix(0);
}


bool dsp::CovarianceMatrixCUDAEngine::hitsContainsZeroes(unsigned int* d_hits, unsigned int hitLength)
{
	int blockDim = 256;
	int gridDim = ceil((float) hitLength / blockDim);

	//Reset d_zeroes to false
	cudaMemset(d_zeroes, 0, sizeof(bool));

	printf("Starting CheckForZeroesKernels: GridDim: %u, BlockDim %u\n", gridDim, blockDim);
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




unsigned int* dsp::CovarianceMatrixCUDAEngine::getHitsPtr(PhaseSeries* phaseSeriesData, CovarianceMatrixResult* covarianceMatrixResult, int freqChan)
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
