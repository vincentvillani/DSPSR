/*
 * CovarianceMatrix.C
 *
 *  Created on: 27/08/2014
 *      Author: vincentvillani
 */


#include "dsp/CovarianceMatrix.h"

dsp::CovarianceMatrix::CovarianceMatrix()
{
	_covarianceMatrixResult = new CovarianceMatrixResult();
	_unloader = NULL;
	_engine = NULL;
}


dsp::CovarianceMatrix::~CovarianceMatrix()
{

#if HAVE_CUDA

	printf("CUDA Destructor called\n");

	if(_covarianceMatrixResult->getNumberOfFreqChans() == 0 || _covarianceMatrixResult->getBinNum() == 0)
	{
		std::cerr << "Something terrible has happened: freqChan: " << _covarianceMatrixResult->getNumberOfFreqChans()
				<< " binNum: " << _covarianceMatrixResult->getBinNum() << std::endl;
		return;
	}

	unsigned int freqChanNum = _covarianceMatrixResult->getNumberOfFreqChans();
	unsigned int binNum = _covarianceMatrixResult->getBinNum();
	unsigned int stokesLength = _covarianceMatrixResult->getStokesLength();
	unsigned int covarianceLength = _covarianceMatrixResult->getCovarianceMatrixLength();

	float* d_outerProducts = _engine->compute_final_covariance_matrices_device(_covarianceMatrixResult);

	//TODO: VINCENT: DEBUG
	//*** DEBUG ****
	float* h_outerProducts = new float[freqChanNum * covarianceLength];
	cudaMemcpy(h_outerProducts, _covarianceMatrixResult->getCovarianceMatrix(0), sizeof(float) * freqChanNum * covarianceLength, cudaMemcpyDeviceToHost);



	//Print out results to a file
	//TODO: VINCENT: DEBUG
	std::stringstream ss;

	cerr << "Before unload" << std::endl;
	//output summed phase series, before normalisation
	_unloader->unload(_covarianceMatrixResult->getPhaseSeries());
	cerr << "After unload" << std::endl;



	//Write out data to a file
	for(int j = 0; j < freqChanNum; ++j)
	{
		//write it out to a file
		ss << "resultMatrixChan" << j << ".txt";
		outputUpperTriangularMatrix(h_outerProducts + (j * covarianceLength), binNum * stokesLength, ss.str());
		ss.str("");
	}



	delete[] h_outerProducts;
	cudaFree(d_outerProducts);
	//delete _unloader; //TODO: VINCENT: IS THIS CORRECT?
	delete _covarianceMatrixResult;


#else


	std::stringstream ss;

	cerr << "Before unload" << std::endl;
	//output summed phase series, before normalisation
	_unloader->unload(_covarianceMatrixResult->getPhaseSeries());
	cerr << "After unload" << std::endl;

	//TODO: VINCENT: DEBUG
	compute_final_covariance_matrices_host();



	unsigned int freqChanNum = _covarianceMatrixResult->getNumberOfFreqChans();
	unsigned int binNum = _covarianceMatrixResult->getBinNum();
	unsigned int stokesLength = _covarianceMatrixResult->getStokesLength();

	//Write out data to a file
	for(int j = 0; j < freqChanNum; ++j)
	{
		//write it out to a file
		ss << "xSquaredCPU" << j << ".txt";
		outputUpperTriangularMatrix(_covarianceMatrixResult->getCovarianceMatrix(j), binNum * stokesLength, ss.str());
		ss.str("");
	}



	delete _unloader; //TODO: VINCENT: IS THIS CORRECT?
	delete _covarianceMatrixResult;

	//if(_engine != NULL)
		//delete _engine;



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
				covariance_matrix_length( phaseSeriesData->get_ndim() * binNum),   phaseSeriesData->get_hits_nchan(), phaseSeriesData);

	}


	//TODO: VINCENT DEBUG
#if !HAVE_CUDA
	printf("StokesLength: %u\n", _covarianceMatrixResult->getStokesLength());
	printf("Value: %f\n", _covarianceMatrixResult->getCovarianceMatrix(0)[0]);
#endif


	if(_engine)
	{
		//compute_covariance_matrix_host(phaseSeriesData);
		#if HAVE_CUDA
			_engine->computeCovarianceMatricesCUDA(phaseSeriesData, _covarianceMatrixResult);
		#endif
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

	float* amps = _covarianceMatrixResult->getAmps();



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
						amps[row] * amps[col];

			}
		}

	}

	_covarianceMatrixResult->getPhaseSeries()->combine(phaseSeriesData); //TODO: VINCENT: DO THIS ON THE GPU
}




void dsp::CovarianceMatrix::norm_stokes_data_host(const float* stokesData, const unsigned int* hits, unsigned int chan)
{
	unsigned int totalLength = _covarianceMatrixResult->getBinNum() * _covarianceMatrixResult->getStokesLength();
	unsigned int stokesLength = _covarianceMatrixResult->getStokesLength();

	float* amps = _covarianceMatrixResult->getAmps();
	float* runningMeanSum = _covarianceMatrixResult->getRunningMeanSum(chan);


	for(unsigned int i = 0; i < totalLength; ++i)
	{

		amps[ i ] = stokesData[ i ] / (hits[ i / stokesLength ]);

		runningMeanSum[ i ] += amps[ i ];

	}
}




void dsp::CovarianceMatrix::compute_final_covariance_matrices_host()
{
	unsigned int freqChanNum = _covarianceMatrixResult->getNumberOfFreqChans();
	unsigned int covarianceMatrixLength = _covarianceMatrixResult->getCovarianceMatrixLength();
	unsigned int unloadCalledNum = _covarianceMatrixResult->getUnloadCallCount();

	//Get the phase series outer product
	float* phaseSeriesOuterProduct =  compute_outer_product_phase_series_host();


	for(int i = 0; i < freqChanNum; ++i)
	{
		float* covarianceMatrix = _covarianceMatrixResult->getCovarianceMatrix(i);

		for(int j = 0; j < covarianceMatrixLength; ++j)
		{

			covarianceMatrix[j] /= unloadCalledNum;
			covarianceMatrix[j] -= phaseSeriesOuterProduct[(i * covarianceMatrixLength) + j];

		}
	}


	/*
	//**** DEBUG ****** TODO:VINCENT: DEBUG

	std::stringstream ss;

	//Write out data to a file
	for(int j = 0; j < freqChanNum; ++j)
	{
		//write it out to a file
		ss << "xMeanCPU" << j << ".txt";
		outputUpperTriangularMatrix(phaseSeriesOuterProduct, _covarianceMatrixResult->getBinNum() * _covarianceMatrixResult->getStokesLength(), ss.str());
		ss.str("");
	}
*/

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
						runningMeanSum[row] * runningMeanSum[col];
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

