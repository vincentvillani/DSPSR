/*
 * CovarianceMatrix.C
 *
 *  Created on: 27/08/2014
 *      Author: vincentvillani
 */


#include "dsp/CovarianceMatrix.h"


dsp::CovarianceMatrix::CovarianceMatrix()
{
	_stokesLength = 4; //TODO: VINCENT - MAKE THIS THIS VARIABLE SOMEHOW


	//initially set pointers to null
	_phaseSeries = NULL;
	_covarianceMatrices = NULL;
	_tempMeanStokesData = NULL;
	_unloader = NULL;



}

dsp::CovarianceMatrix::~CovarianceMatrix()
{
	delete _phaseSeries; //TODO: VINCENT: IS THIS CORRECT?
	delete _unloader; //TODO: VINCENT: IS THIS CORRECT?
	delete [] _tempMeanStokesData;

	for(int i = 0; i < _freqChanNum; ++i)
		delete [] _covarianceMatrices[i];

	delete [] _covarianceMatrices;

}


void dsp::CovarianceMatrix::unload(const PhaseSeries* phaseSeriesData)
{
	printf("GOT MY PHASE SERIES INTEGRATED DATA\n");

	printf("Num of Phase Bins: %u\n", phaseSeriesData->get_nbin());
	printf("Number of channels in the hit array: %u\n", phaseSeriesData->get_hits_nchan() );

	//Num bins == 0 the first time this is called for some reason?
	unsigned int binNum = phaseSeriesData->get_nbin();
	//unsigned int numChannels = phaseSeriesData->get_hits_nchan(); //number of freq channels
	bool firstIteration = false;



	//first time this is called numBins may be zero
	if(binNum == 0)
		return;

	//Allocate memory on first use, once we know the amount of memory required
	if(_covarianceMatrices == NULL && binNum != 0)
	{


		//This is the first iteration, we can take shortcuts
		firstIteration = true;

		_binNum = binNum;
		_freqChanNum = phaseSeriesData->get_hits_nchan();
		_covarianceMatrixLength = covariance_matrix_length(_binNum) * _stokesLength;

		//Allocate memory to store the covarianceMatrix
		//upper triangle * 4 stokes vector elements
		_covarianceMatrices = new float*[_freqChanNum]; //allocate a pointer for each channel
		//_summedMeanStokesDatas = new float*[numChannels]; //allocate a pointer for each channel

		for(int i = 0; i < _freqChanNum; ++i)
		{
			//Assign the amount of memory needed for a covariance matrix in each freq channel
			_covarianceMatrices[i] = new float[ _covarianceMatrixLength ];
			memset(_covarianceMatrices[i], 0, sizeof(float) * _covarianceMatrixLength); //Set all the values to zero

			//Assign the amount of memory needed for the running total of mean stokes data in each freq channel
			//_summedMeanStokesDatas[i] = new float[ numBins * _stokesLength ];
		}


		//allocate scratch space for temporary data
		_tempMeanStokesData = new float[_binNum * _stokesLength];

		//clone the first phase series
		//_phaseSeries = new PhaseSeries(*phaseSeriesData);

	}


	// ------ HOST COMPUTE CODE ----------



	//take shortcuts
	if(firstIteration)
	{

		//For each channel
		for(unsigned int channel = 0; channel < _freqChanNum; ++channel)
		{

			//------- AMPLITUDE DATA ------


			//TODO: VINCENT: FIGURE OUT IF THE END OF STOKES-I IS NEXT TO THE START OF STOKES-Q
			const float* stokesI = phaseSeriesData->get_datptr(channel, 0); //Get a pointer to all the I stokes values
			const float* stokesQ = phaseSeriesData->get_datptr(channel, 1); //Get a pointer to all the Q stokes values
			const float* stokesU = phaseSeriesData->get_datptr(channel, 2); //Get a pointer to all the U stokes values
			const float* stokesV = phaseSeriesData->get_datptr(channel, 3); //Get a pointer to all the V stokes values

			const unsigned int* hits = phaseSeriesData->get_hits(channel); //Get a pointer to the hit data



			//normalise the stokes data for this freq channel
			mean_stokes_data_host(stokesI, hits, 0);
			mean_stokes_data_host(stokesQ, hits, _binNum);
			mean_stokes_data_host(stokesU, hits, _binNum * 2);
			mean_stokes_data_host(stokesV, hits, _binNum * 3);


			//compute the covariance matrix
			compute_covariance_matrix_host(channel);


			// --------

		}


	}

	printf("FINISHED UNLOAD\n\n\n");

}


void dsp::CovarianceMatrix::compute_covariance_matrix_host(unsigned int freqChan)
{
	for(int row = 0; row < _binNum; ++row)
	{
		for(int col = row; col < _binNum; ++col)
		{
			_covarianceMatrices[freqChan][ (row * _binNum + col) - covariance_matrix_length(row) ] +=
					_tempMeanStokesData[row] * _tempMeanStokesData[col];
		}
	}


}


void dsp::CovarianceMatrix::mean_stokes_data_host(const float* stokesData, const unsigned int* hits, unsigned int offset)
{
	for(int i = 0; i < _binNum; ++i)
	{
		_tempMeanStokesData[offset + i] = stokesData[i] / hits[ i / 4 ];
	}
}


unsigned int dsp::CovarianceMatrix::covariance_matrix_length(const unsigned int numBin)
{
	return (numBin * (numBin + 1)) / 2;
}


void dsp::CovarianceMatrix::set_unloader(PhaseSeriesUnloader* unloader)
{
	_unloader = unloader;
}

