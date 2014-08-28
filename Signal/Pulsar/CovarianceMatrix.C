/*
 * CovarianceMatrix.C
 *
 *  Created on: 27/08/2014
 *      Author: vincentvillani
 */


#include "dsp/CovarianceMatrix.h"


dsp::CovarianceMatrix::CovarianceMatrix()
{
	printf("COVARIANCE MATRIX CREATED!!!!!!\n");
	_covarianceMatrix = NULL;
	_ampsData = NULL;
	_hitsData = NULL;
}


void dsp::CovarianceMatrix::unload(const PhaseSeries* phaseSeriesData)
{
	printf("GOT MY PHASE SERIES INTEGRATED DATA\n");

	printf("Num of Phase Bins: %u\n", phaseSeriesData->get_nbin());
	printf("Number of channels in the hit array: %u\n", phaseSeriesData->get_hits_nchan ());

	//Num bins == 0 the first time this is called for some reason?
	unsigned int numBins = phaseSeriesData->get_nbin();
	unsigned int numChannels = phaseSeriesData->get_hits_nchan (); //number of freq channels
	bool firstIteration = false;


	//Allocate memory on first use, once we know the amount of memory required
	if(_covarianceMatrix == NULL && numBins != 0)
	{
		//This is the first iteration, we can take shortcuts
		firstIteration = true;

		//TODO: VINCENT: ADD OTHER CHANNELS?
		//Allocate memory to store the covarianceMatrix
		//bins^2 * 4 stokes vector elements
		_covarianceMatrix = new float[ numBins * numBins * 4 ];

		//Allocate memory to store the summed amps data
		//bins * 4 stokes vector elements?
		_ampsData = new float[numBins * 4];

		//Allocate memory to store summed hits data
		//bins * 4 stokes vector elements?
		_hitsData = new unsigned int[numBins];
	}


	//take shortcuts
	if(firstIteration)
	{
		//TODO: VINCENT: ACTUALLY ADD MEMORY STRUCTURES TO HANDLE MORE THAN ONE FREQ CHANNEL
		//For each channel
		for(int i = 0; i < numChannels; ++i)
		{

			//------- AMPLITUDE DATA ------

			//TODO: VINCENT: FIGURE OUT IF THE END OF STOKES-I IS NEXT TO THE START OF STOKES-Q
			const float* stokesI = phaseSeriesData->get_datptr(i, 0); //Get a pointer to all the I stokes values
			const float* stokesQ = phaseSeriesData->get_datptr(i, 1); //Get a pointer to all the Q stokes values
			const float* stokesU = phaseSeriesData->get_datptr(i, 2); //Get a pointer to all the U stokes values
			const float* stokesV = phaseSeriesData->get_datptr(i, 3); //Get a pointer to all the V stokes values

			//Simply copy all the data across, no need to compute or call kernels
			memcpy(_ampsData, stokesI, sizeof(float) * numBins);
			memcpy(_ampsData + numBins, stokesQ, sizeof(float) * numBins);
			memcpy(_ampsData + (numBins * 2), stokesU, sizeof(float) * numBins);
			memcpy(_ampsData + (numBins * 3), stokesV, sizeof(float) * numBins);


			//------- HIT DATA -----

			const unsigned int* hits = phaseSeriesData->get_hits(i); //Get a pointer to the hit data

			//Simply copy all the data across, no need to compute or call kernels
			memcpy(_hitsData, hits, sizeof(unsigned int) * numBins);


		}


	}






}

