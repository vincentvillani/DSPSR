/*
 * CovarianceMatrix.C
 *
 *  Created on: 27/08/2014
 *      Author: vincentvillani
 */


#include "dsp/CovarianceMatrix.h"

dsp::CovarianceMatrix::CovarianceMatrix(bool useCuda)
{
	_covarianceMatrixResult = new CovarianceMatrixResult(useCuda);
	_unloader = NULL;
	_engine = NULL;
	_useCuda = useCuda;
}


dsp::CovarianceMatrix::~CovarianceMatrix()
{

#if HAVE_CUDA

	if(_useCuda)
	{
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

		//cerr << "Before unload" << std::endl;
		//output summed phase series, before normalisation
		//_unloader->unload(_covarianceMatrixResult->getPhaseSeries());
		//cerr << "After unload" << std::endl;



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
	}


#else


	std::stringstream ss;

	//output summed phase series, before normalisation
	_unloader->unload(_covarianceMatrixResult->getPhaseSeries());

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

	delete _covarianceMatrixResult;


#endif

	//printf("DESTRUCTOR ENDED\n");

}




void dsp::CovarianceMatrix::unload(const PhaseSeries* phaseSeriesData)
{

	printf("\n\nCOVARIANCEMATRIX PS\n");
	phaseSeriesData->print();

	unsigned int binNum = phaseSeriesData->get_nbin();

	/*
	std::cerr << "dsp::CovarianceMatrix::unload Freq Chans: " << phaseSeriesData->get_nchan() << ", Binsize: "
			<< binNum << ", NPol: " << phaseSeriesData->get_npol() << ", NDim: " << phaseSeriesData->get_ndim()
			<< std::endl << ", hit channels: " << phaseSeriesData->get_hits_nchan() <<  ", State: " << phaseSeriesData->get_state()
			<< std::endl;
	*/


	//first time this is called numBins may be zero
	if(binNum == 0)
		return;

	//Allocate memory on first use, once we know the amount of memory required
	if(!_covarianceMatrixResult->hasBeenSetup())
	{

		//Setup the covariance matrix result object
		_covarianceMatrixResult->setup(binNum, phaseSeriesData->get_nchan(), phaseSeriesData->get_ndim(),
				covariance_matrix_length( phaseSeriesData->get_ndim() * binNum),   phaseSeriesData->get_hits_nchan(), phaseSeriesData);

	}


	/*
#if !HAVE_CUDA
	printf("StokesLength: %u\n", _covarianceMatrixResult->getStokesLength());
	printf("Value: %f\n", _covarianceMatrixResult->getCovarianceMatrix(0)[0]);
#endif
*/


	if(_engine)
	{
		//compute the outer product using the GPU
		#if HAVE_CUDA
			_engine->computeCovarianceMatricesCUDA(phaseSeriesData, _covarianceMatrixResult);
		#endif
	}
	else
	{
		//compute the outer product using the CPU
		compute_covariance_matrix_host(phaseSeriesData);
	}

	//Add one to the unloadCallCount
	_covarianceMatrixResult->iterateUnloadCallCount();
}



void dsp::CovarianceMatrix::compute_covariance_matrix_host(const PhaseSeries* phaseSeriesData)
{

	fprintf(stderr, "CPU: PS amps value: %f\n", *phaseSeriesData->get_datptr(0, 0));

	unsigned int chanNum = _covarianceMatrixResult->getNumberOfFreqChans();
	unsigned int binNum = _covarianceMatrixResult->getBinNum();
	unsigned int stokesLength = _covarianceMatrixResult->getStokesLength();
	unsigned int hitChanNum = _covarianceMatrixResult->getNumberOfHitChans();

	//Amp's values are set in the norm_stokes_data_host() method
	//It's basically used as scratch space for the normalised amplitude data
	//Only has enough memory allocated to hold one freq channels worth of amp data
	//I.E StokesLength * BinNum == ampsLength
	float* amps = _covarianceMatrixResult->getAmps();



	//Check the hits array for 0 values, if they exist throw this whole Phase Series away and exit
	for(unsigned int currentHitChan = 0; currentHitChan < hitChanNum; ++currentHitChan)
	{
		//Get a pointer to this channels hits data
		//If there is only one hits array then this
		//will return the same one each time
		const unsigned int* hits = getHitsPtr(phaseSeriesData, currentHitChan);

		for(unsigned int i = 0; i < binNum; ++i)
		{
			//Zero value detected, abort this calculation
			if(hits[i] == 0)
				return;
		}
	}


	//Go through each channel and first normalise the amps values
	//then compute the outer product and add it to the existing values
	for(unsigned int channel = 0; channel < chanNum; ++channel)
	{

		//Get a pointer to this channels amp data from the PhaseSeries
		const float* stokes = phaseSeriesData->get_datptr(channel, 0);

		//Get a pointer to this channels hits data
		//If there is only one hits array then this
		//will return the same one each time
		const unsigned int* hits = getHitsPtr(phaseSeriesData, channel);

		//A place to store the outer product result
		float* covarianceMatrix =  _covarianceMatrixResult->getCovarianceMatrix(channel);

		//normalise the amps data for this freq channel
		//by dividing it by its corresponding hit value
		//Also adds the normalised amps to a running total
		//to be used later to calcualte the covariance matrix
		norm_stokes_data_host(stokes, hits, channel);



		//rowLength == colLength
		unsigned int rowLength = binNum * stokesLength;

		//Compute the outer product and add it do the previously calculated values
		for(unsigned int row = 0; row < rowLength; ++row)
		{
			for(unsigned int col = row; col < rowLength; ++col)
			{
				//Compute the correct index to add and store the outer product result in.
				//Calculation is basically:
				//( (element number in a full matrix) - (number of zero values due to it being an upper triangular matrix))
				covarianceMatrix[ (row * rowLength + col) - covariance_matrix_length(row) ] +=
						amps[row] * amps[col];

			}
		}

	}

	//Combine this phase series with the previous calculated phase series
	_covarianceMatrixResult->getPhaseSeries()->combine(phaseSeriesData);
}




void dsp::CovarianceMatrix::norm_stokes_data_host(const float* stokesData, const unsigned int* hits, unsigned int chan)
{
	unsigned int stokesLength = _covarianceMatrixResult->getStokesLength();
	unsigned int totalLength = _covarianceMatrixResult->getBinNum() * stokesLength;


	//Get a pointer to scratch space to store the normalised amps
	float* amps = _covarianceMatrixResult->getAmps();

	//Get a pointer to add the normalised amps values to
	//This will be used later to compute the covariance matrix
	float* runningMeanSum = _covarianceMatrixResult->getRunningMeanSum(chan);



	for(unsigned int i = 0; i < totalLength; ++i)
	{
		//Calculate the normalised amps values and store it in the scratch space
		amps[i] = stokesData[i] / (hits[i / stokesLength]);

		//Add it to the running mean
		runningMeanSum[i] += amps[i];
	}
}




void dsp::CovarianceMatrix::compute_final_covariance_matrices_host()
{
	unsigned int freqChanNum = _covarianceMatrixResult->getNumberOfFreqChans();
	unsigned int covarianceMatrixLength = _covarianceMatrixResult->getCovarianceMatrixLength();
	unsigned int unloadCalledNum = _covarianceMatrixResult->getUnloadCallCount();

	//Calculate and get a pointer to the Phase Series Outer Product
	float* phaseSeriesOuterProduct =  compute_outer_product_phase_series_host();


	for(int i = 0; i < freqChanNum; ++i)
	{
		//when this is called, covariance matrix should contain the outer products of the
		//normalised amps values
		float* covarianceMatrix = _covarianceMatrixResult->getCovarianceMatrix(i);

		//Divide by the number of times unload() was called
		for(int j = 0; j < covarianceMatrixLength; ++j)
		{
			//Divide by the number of times unload() was called
			covarianceMatrix[j] /= unloadCalledNum;

			//subtracting the normalised amps outer product (covarianceMatrix) from the phase series outer (the outer product of the running mean)
			//should result in the final 'covariance matrix'
			//I really should pick better names for these...
			covarianceMatrix[j] -= phaseSeriesOuterProduct[(i * covarianceMatrixLength) + j];

		}
	}

	//Free the previously allocated memory in compute_outer_product_phase_series_host()
	delete[] phaseSeriesOuterProduct;
}




float* dsp::CovarianceMatrix::compute_outer_product_phase_series_host()
{
	//Number of times unload() was called and a phase series was passed
	//to the covariance matrix object
	unsigned int unloadCallCount = _covarianceMatrixResult->getUnloadCallCount();
	unsigned int freqChanNum = _covarianceMatrixResult->getNumberOfFreqChans();
	unsigned int covarianceLength = _covarianceMatrixResult->getCovarianceMatrixLength();
	unsigned int ampsLength = _covarianceMatrixResult->getBinNum() * _covarianceMatrixResult->getStokesLength();


	//allocate enough space to store the outer product
	float* outerProduct = new float [freqChanNum * covarianceLength];


	//For each freq channel
	for(unsigned int channel = 0; channel < freqChanNum; ++channel)
	{
		//Get a pointer to the running mean data
		float* runningMeanSum = _covarianceMatrixResult->getRunningMeanSum(channel);

		//divide running mean sum by number of times unload() was called
		for(unsigned int i = 0; i < ampsLength; ++i)
			runningMeanSum[i] /= unloadCallCount;


		//Do the outer product calculation
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
	//If there is only one hit channel, return it every time
	if(_covarianceMatrixResult->getNumberOfHitChans() == 1)
		return phaseSeriesData->get_hits(0);

	//Return the hits pointer using the freq channel
	else
		return phaseSeriesData->get_hits(freqChan);
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
	float* fullMatrix = new float[rowLength * rowLength];

	if(fullMatrix == NULL)
	{
		printf("MALLOC ERROR\n");
		return NULL;
	}

	//For each row
	for(unsigned int row = 0; row < rowLength; ++row)
	{
		// ---- FULL MATRIX INDEXES ----

		//Compute the diagonalIdx
		unsigned int diagonalIndex = (row * rowLength) + row;

		// ---- TRI MATRIX INDEXES ----
		unsigned int triDiagonalIndex = diagonalIndex - ( (row * (row + 1)) / 2);

		unsigned int indexOffset = 0;
		//print down the corresponding row and column
		for(unsigned int printIdx = row; printIdx < rowLength; ++printIdx)
		{
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

