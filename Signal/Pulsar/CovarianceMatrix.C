/*
 * CovarianceMatrix.C
 *
 *  Created on: 27/08/2014
 *      Author: vincentvillani
 */


#include "dsp/CovarianceMatrix.h"

dsp::CovarianceMatrix::CovarianceMatrix()
{
	//initially set pointers to null
	_phaseSeries = NULL;
	_covarianceMatrixResult = new CovarianceMatrixResult();
	_unloader = NULL;
	_engine = NULL;
}


dsp::CovarianceMatrix::~CovarianceMatrix()
{

#if HAVE_CUDA



#else

	//TODO: VINCENT: DEBUG
	std::stringstream ss;

	cerr << "Before unload" << std::endl;
	//output summed phase series, before normalisation
	_unloader->unload(_phaseSeries);
	cerr << "After unload" << std::endl;

	compute_final_covariance_matrices_host();


	unsigned int freqChanNum = _covarianceMatrixResult->getNumberOfFreqChans();
	unsigned int binNum = _covarianceMatrixResult->getBinNum();
	unsigned int stokesLength = _covarianceMatrixResult->getStokesLength();

	//Write out data to a file
	for(int j = 0; j < freqChanNum; ++j)
	{
		//write it out to a file
		ss << "resultMatrixChan" << j << ".txt";
		outputUpperTriangularMatrix(_covarianceMatrixResult->getCovarianceMatrix(j), binNum * stokesLength, ss.str());
		ss.str("");
	}



	delete _phaseSeries; //TODO: VINCENT: IS THIS CORRECT?
	delete _unloader; //TODO: VINCENT: IS THIS CORRECT?
	delete _covarianceMatrixResult;

	if(_engine != NULL)
		delete _engine;



#endif

	printf("DESTRUCTOR ENDED\n");

}




void dsp::CovarianceMatrix::unload(const PhaseSeries* phaseSeriesData)
{

#ifdef HAVE_CUDA

	printf("Has cuda!\n");
#else
	printf("Does not have cuda\n");

#endif



	unsigned int binNum = phaseSeriesData->get_nbin();

	std::cerr << "dsp::CovarianceMatrix::unload Freq Chans: " << phaseSeriesData->get_nchan() << ", Binsize: "
			<< binNum << ", NPol: " << phaseSeriesData->get_npol() << ", NDim: " << phaseSeriesData->get_ndim()
			<< std::endl << ", hit channels: " << phaseSeriesData->get_hits_nchan() <<  ", State: " << phaseSeriesData->get_state()
			<< std::endl;


	//first time this is called numBins may be zero
	if(binNum == 0)
		return;

	//Allocate memory on first use, once we know the amount of memory required
	if(!_covarianceMatrixResult->hasBeenSetup())
	{

		//Setup the covariance matrix
		_covarianceMatrixResult->setup(binNum, phaseSeriesData->get_nchan(), phaseSeriesData->get_ndim(),
				covariance_matrix_length( phaseSeriesData->get_ndim() * binNum),   phaseSeriesData->get_hits_nchan());

		//clone the first phase series
		_phaseSeries = new PhaseSeries(*phaseSeriesData);

	}


	printf("StokesLength: %u\n", _covarianceMatrixResult->getStokesLength());
	printf("Value: %f\n", _covarianceMatrixResult->getCovarianceMatrix(0)[0]);

	if(_engine)
	{
		/*
		 * void CovarianceMatrixCUDAEngine::computeCovarianceMatrix(float* d_result,
	const float* h_amps, float* d_amps, unsigned int ampsLength,
	const unsigned int* h_hits, unsigned int* d_hits, unsigned int hitsLength,
	unsigned int stokesLength, unsigned int blockDim2D)
		 */

		for(int i = 0; i < _covarianceMatrixResult->getNumberOfFreqChans(); ++i)
		{
			//_engine->computeCovarianceMatrix(
			//		_covarianceMatrixResult->getCovarianceMatrix(i),
			//		phaseSeriesData->get_datptr(i, 0), _covarianceMatrixResult->)
		}

		//_engine->compute_final_covariance_matrices_device(_covarianceMatrixResult->getCovarianceMatrix(0), )
	}
	else
	{
		//compute the covariance matrix
		compute_covariance_matrix_host(phaseSeriesData);
	}

	_covarianceMatrixResult->iterateUnloadCallCount();

	printf("FINISHED UNLOAD\n\n\n");
}



void dsp::CovarianceMatrix::compute_covariance_matrix_host(const PhaseSeries* phaseSeriesData)
{

	unsigned int chanNum = _covarianceMatrixResult->getNumberOfFreqChans();
	unsigned int binNum = _covarianceMatrixResult->getBinNum();
	unsigned int stokesLength = _covarianceMatrixResult->getStokesLength();
	unsigned int hitChanNum = _covarianceMatrixResult->getNumberOfHitChans();

	float* tempMeanStokesData = _covarianceMatrixResult->getTempMeanStokesData();



	//check for no hits, if they exist discard this whole phase-series

	//check for no hits, if they exist discard this whole phase-series
	for(unsigned int currentHitChan = 0; currentHitChan < hitChanNum; ++currentHitChan)
	{
		const unsigned int* hits = getHitsPtr(phaseSeriesData, currentHitChan);

		for(unsigned int i = 0; i < binNum; ++i)
		{
			if(hits[i] == 0)
				return;
		}
	}



	for(unsigned int channel = 0; channel < chanNum; ++channel)
	{
		//AMPLITUDE DATA
		//IQUV, IQUV, IQUV etc etc
		const float* stokes = phaseSeriesData->get_datptr(channel, 0); //Get a pointer to the amps data
		const unsigned int* hits = getHitsPtr(phaseSeriesData, channel);
		float* covarianceMatrix =  _covarianceMatrixResult->getCovarianceMatrix(channel);

		//normalise the stokes data for this freq channel
		norm_stokes_data_host(stokes, hits, channel);


		//Compute the covariance matrix
		//ColLength == rowLength
		unsigned int rowLength = binNum * stokesLength;

		for(unsigned int row = 0; row < rowLength; ++row)
		{
			for(unsigned int col = row; col < rowLength; ++col)
			{
				covarianceMatrix[ (row * rowLength + col) - covariance_matrix_length(row) ] +=
						tempMeanStokesData[row] * tempMeanStokesData[col];

			}
		}

	}

	_phaseSeries->combine(phaseSeriesData); //TODO: VINCENT: DO THIS ON THE GPU
}




void dsp::CovarianceMatrix::norm_stokes_data_host(const float* stokesData, const unsigned int* hits, unsigned int chan)
{
	unsigned int totalLength = _covarianceMatrixResult->getBinNum() * _covarianceMatrixResult->getStokesLength();
	unsigned int stokesLength = _covarianceMatrixResult->getStokesLength();

	float* tempMeanStokesData = _covarianceMatrixResult->getTempMeanStokesData();
	float* runningMeanSum = _covarianceMatrixResult->getRunningMeanSum(chan);


	for(unsigned int i = 0; i < totalLength; ++i)
	{

		tempMeanStokesData[ i ] = stokesData[ i ] / (hits[ i / stokesLength ]);

		runningMeanSum[ i ] += tempMeanStokesData[ i ];

	}
}




void dsp::CovarianceMatrix::compute_final_covariance_matrices_host()
{
	unsigned int freqChanNum = _covarianceMatrixResult->getNumberOfFreqChans();
	unsigned int covarianceMatrixLength = _covarianceMatrixResult->getCovarianceMatrixLength();
	unsigned int unloadCalledNum = _covarianceMatrixResult->getUnloadCallCount();

	//Get the phase series outer product
	float* phaseSeriesOuterProduct =  compute_outer_product_phase_series_host(); //compute_outer_product_phase_series_host();


	for(int i = 0; i < freqChanNum; ++i)
	{
		float* covarianceMatrix = _covarianceMatrixResult->getCovarianceMatrix(i);

		for(int j = 0; j < covarianceMatrixLength; ++j)
		{

			covarianceMatrix[j] /= unloadCalledNum;
			covarianceMatrix[j] -= phaseSeriesOuterProduct[(i * covarianceMatrixLength) + j];

		}
	}

	delete[] phaseSeriesOuterProduct;
}




float* dsp::CovarianceMatrix::compute_outer_product_phase_series_host()
{
	unsigned int unloadCallCount = _covarianceMatrixResult->getUnloadCallCount();
	unsigned int freqChanNum = _covarianceMatrixResult->getNumberOfFreqChans();
	unsigned int covarianceLength = _covarianceMatrixResult->getCovarianceMatrixLength();
	unsigned int ampsLength = _covarianceMatrixResult->getBinNum() * _covarianceMatrixResult->getStokesLength();


	float* outerProduct = new float [freqChanNum * covarianceLength];


	//For each freq channel
	for(unsigned int channel = 0; channel < freqChanNum; ++channel)
	{

		float* runningMeanSum = _covarianceMatrixResult->getRunningMeanSum(channel);

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
				outerProduct[ (channel * covarianceLength) +  ((row * ampsLength + col) - covariance_matrix_length(row)) ] =
						runningMeanSum[row] * runningMeanSum[col]; //amps[row] * amps[col];
			}
		}

	}


	return outerProduct;
}




const unsigned int* dsp::CovarianceMatrix::getHitsPtr(const PhaseSeries* phaseSeriesData, int freqChan)
{
	//return the only channel
	if(_covarianceMatrixResult->getNumberOfHitChans() == 1)
		return phaseSeriesData->get_hits(0);
	else
		return phaseSeriesData->get_hits(freqChan); //Return the hits pointer using the freq channel
}




void dsp::CovarianceMatrix::set_engine(CovarianceMatrixCUDAEngine* engine)
{
	_engine = engine;
	//_memory = Memory::get_manager();
}




/*
#if HAVE_CUDA

void dsp::CovarianceMatrix::setup_device(unsigned int chanNum, unsigned int hitChanNum, unsigned int binNum, unsigned int nPol, unsigned int nDim)
{

	_binNum = binNum;
	_freqChanNum = chanNum;
	_stokesLength = nDim; //TODO: VINCENT: CORRECT??
	_hitChanNum = hitChanNum;

	printf("Allocating device memory\n");

	//Check for correct channels
	if(hitChanNum != 1 && hitChanNum != chanNum)
	{
		//TODO: VINCENT: THROW AN EXCEPTION
		printf("INVALID NUMBER OF HIT CHANNELS\n");
		exit(2);
	}



	_covarianceMatrixLength = covariance_matrix_length(_binNum * _stokesLength);

	//Allocate paged locked host memory for the pointer
	cudaMallocHost(&_covarianceMatrices, sizeof(float*) * _freqChanNum);


	for(int i = 0; i < _freqChanNum; ++i)
	{
		//Assign the amount of paged locked memory needed for a covariance matrix in each freq channel
		cudaMallocHost(&(_covarianceMatrices[i]), sizeof(float) * _covarianceMatrixLength);

		//Set all the values to zero
		memset(_covarianceMatrices[i], 0, sizeof(float) * _covarianceMatrixLength);
	}


	cudaMalloc(&_d_amps, sizeof(float) * _binNum * _stokesLength);
	cudaMalloc(&_d_hits, sizeof(unsigned int) * _binNum );


	size_t totalResultByteNum = sizeof(float) * _covarianceMatrixLength  * _freqChanNum;

	//Allocate space for all result vectors
	cudaMalloc(&_d_resultVector, totalResultByteNum);

	//Set all bytes to zero
	cudaMemset(_d_resultVector, 0, totalResultByteNum);

	//TODO: VINCENT: DEBUG
	cudaError_t error = cudaDeviceSynchronize();
	if(error != cudaSuccess)
	{
		printf("CUDA ERROR: %s\n", cudaGetErrorString(error));
	}
}



void dsp::CovarianceMatrix::compute_covariance_matrix_device(const PhaseSeries* phaseSeriesData)
{

	printf("FreqChanNum: %d\n", _freqChanNum);

	for(int channel = 0; channel < _freqChanNum; ++channel)
	{
		printf("\nFreq %d\n", channel);

		const float* h_amps = phaseSeriesData->get_datptr(channel, 0);
		//const unsigned int* h_hits = phaseSeriesData->get_hits(0); //TODO: VINCENT, THIS COULD BE THE SOURCE OF ERRORS LATER

		const unsigned int* h_hits = getHitsPtr(phaseSeriesData, channel);

		computeCovarianceMatrixCUDAEngine (_d_resultVector, channel * _covarianceMatrixLength,
			h_amps, _d_amps, _binNum * _stokesLength,
			h_hits, _d_hits, _binNum, _stokesLength, phaseSeriesData->get_scale() );
	}

	_phaseSeries->combine(phaseSeriesData); //TODO: VINCENT: DO THIS ON THE GPU

}


void dsp::CovarianceMatrix::copyAndPrint(float* deviceData, int arrayLength, int rowLength)
{
	float* hostData = new float [arrayLength];
	cudaMemcpy(hostData, deviceData, sizeof(float) * arrayLength, cudaMemcpyDeviceToHost);
	printUpperTriangularMatrix(hostData, rowLength, true);
}


#endif
*/



unsigned int dsp::CovarianceMatrix::covariance_matrix_length(unsigned int numBin)
{
	return (numBin * (numBin + 1)) / 2;
}




void dsp::CovarianceMatrix::set_unloader(PhaseSeriesUnloader* unloader)
{
	_unloader = unloader;
}



//TODO: VINCENT, EVERYTHING BELOW HERE IS DEBUG ONLY



void dsp::CovarianceMatrix::printUpperTriangularMatrix(float* result, int rowLength, bool genFile)
{
	int numZeros = 0;
	int iterator = 0;

	if(genFile)
	{
		FILE* file = fopen("/mnt/home/vvillani/DSPSR/resultMatrix.txt", "w");

		//for every row
		for(int i = 0; i < rowLength; ++i)
		{
			//print preceding zeros
			for(int j = 0; j < numZeros; ++j)
			{
				fprintf(file, "0, ");
			}

			//print array values
			for(int k = 0; k < rowLength - numZeros; ++k)
			{
				fprintf(file, "%f, ", result[iterator]);
				++iterator;
			}

			fprintf(file, "\n");
			numZeros++;
		}

		fclose(file);

	}
}





float* dsp::CovarianceMatrix::convertToSymmetric(float* upperTriangle, unsigned int rowLength)
{
	//rowLength == colLength

	//printf("ROW LENGTH: %u\n", rowLength);

	float* fullMatrix = new float[rowLength * rowLength];

	if(fullMatrix == NULL)
	{
		printf("MALLOC ERROR\n");
		exit(5);
	}

	//For each row
	for(unsigned int row = 0; row < rowLength; ++row)
	{
		// ---- FULL MATRIX INDEXES ----

		//Compute the diagonalIdx
		unsigned int diagonalIndex = (row * rowLength) + row;

        //printf("DiagonalIndex: %d\n", diagonalIndex);

		// ---- TRI MATRIX INDEXES ----
		unsigned int triDiagonalIndex = diagonalIndex - ( (row * (row + 1)) / 2);

		unsigned int indexOffset = 0;
		//print down the corresponding row and column
		for(unsigned int printIdx = row; printIdx < rowLength; ++printIdx)
		{
			//TODO: VINCENT DEBUG
            if(diagonalIndex + indexOffset > (rowLength * rowLength) - 1)
			{
				printf("INVALID ROW INDEX!!\n");
                printf("diagonalIndex: %d\n", diagonalIndex);
                printf("ROW INDEX: %d\n", diagonalIndex + indexOffset);

			}

            if(diagonalIndex + (rowLength * indexOffset) > (rowLength * rowLength) - 1)
            {
                printf("INVALID COL INDEX!!\n");
                printf("Outer loop: %d, Inner loop: %d\n", row, printIdx);
                printf("diagonalIndex: %d\n", diagonalIndex);
                printf("COL INDEX: %d\n\n", diagonalIndex + (rowLength * indexOffset));
            }

			unsigned int upperTriIndex = triDiagonalIndex + indexOffset;

			//place in row
			fullMatrix[diagonalIndex + indexOffset] = upperTriangle[upperTriIndex];

            if(diagonalIndex + (rowLength * indexOffset) >= (rowLength * rowLength) - 1)
                continue;

			//place in col
			fullMatrix[diagonalIndex + (rowLength * indexOffset)] = upperTriangle[upperTriIndex];
			++indexOffset;
		}
	}

	return fullMatrix;
}



void dsp::CovarianceMatrix::printSymmetricMatrix(float* symmetricMatrix, int rowLength, bool genFile)
{
	//rowLength == colLength

	if(genFile)
	{
		FILE* file = fopen("/mnt/home/vvillani/DSPSR/symmetricMatrix.txt", "w");

		for(int i = 0; i < rowLength * rowLength; ++i)
		{
			if(i != 0 && (i % rowLength) == 0)
				fprintf(file, "\n");

			fprintf(file, "%f ", symmetricMatrix[i]);
		}

	    fclose(file);
	}
	else
	{
		for(int i = 0; i < rowLength * rowLength; ++i)
		{
			if(i != 0 && (i % rowLength) == 0)
				printf("\n");

			printf("%f, ", symmetricMatrix[i]);
		}

	    printf("\n\n");
	}


}


void dsp::CovarianceMatrix::outputSymmetricMatrix(float* symmetricMatrix, unsigned int rowLength, std::string filename)
{
	FILE* file = fopen(filename.c_str(), "w");

	for(int i = 0; i < rowLength * rowLength; ++i)
	{
		if((i % rowLength) == 0 && i != 0)
			fprintf(file, "\n");

		fprintf(file, "%f ", symmetricMatrix[i]);
	}

	fclose(file);
}



void dsp::CovarianceMatrix::outputUpperTriangularMatrix(float* result, unsigned int rowLength, std::string filename)
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


bool dsp::CovarianceMatrix::engine_set()
{
	if(_engine == NULL)
		return false;
	else
		return true;
}

