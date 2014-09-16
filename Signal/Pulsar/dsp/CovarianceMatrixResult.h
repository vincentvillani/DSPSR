/*
 * CovarianceMatrixResult.h
 *
 *  Created on: 10/09/2014
 *      Author: vincentvillani
 */

#ifndef COVARIANCEMATRIXRESULT_H_
#define COVARIANCEMATRIXRESULT_H_

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include <iostream>
#include <cstring>
#include "dsp/DataSeries.h"
#include "dsp/PhaseSeries.h"
//#include "memory.h"
//#include "ReferenceTo.h"
//#include "ReferenceTo.h"

#if HAVE_CUDA
#include <cuda_runtime.h>
#endif


namespace dsp
{

	class CovarianceMatrixResult : public DataSeries
	{

	private:
		unsigned int _binNum; //number of bins
		unsigned int _freqChanNum; //number of frequency channels
		unsigned int _stokesLength; //Length of the stokes vector
		unsigned int _covarianceMatrixLength; //Array length of a covariance matrix
		unsigned int _hitChanNum; //Number of hit channels
		unsigned int _unloadCalledNum; //Number of times unload has been called

		PhaseSeries* _phaseSeries;

		float* _amps;
		unsigned int* d_hits;

		float* d_outerProducts; //Pointer to device memory to store summed outer products
		unsigned int _outerProductsLength; //Total outer product length

		float* _runningMeanSum; //Running total of the mean for each freq channel
		unsigned int _runningMeanSumLength; //total runningMeanSumLength

		//float* _tempNormalisedAmps; //scratch space for ONE covariance matrix (length == covarianceMatrixLength)



		//TODO: VINCENT: DO THIS PROPERLY - REMOVE THIS
		bool _useCUDA;
		bool _setup;



	public:
		CovarianceMatrixResult();
		~CovarianceMatrixResult();

		//TODO: VINCENT: DO THIS PROPERLY
		CovarianceMatrixResult* null_clone() const {return NULL;}
		CovarianceMatrixResult* clone() const {return NULL;}

		void setup(unsigned int binNum, unsigned int freqChanNum, unsigned int stokesLength,
					unsigned int covarianceMatrixLength, unsigned int hitChannelNumber, const PhaseSeries* ps);




		//Getters
		float* getCovarianceMatrix(unsigned int channelOffset);
		float* getRunningMeanSum(unsigned int channelOffset);
		//float* getTempMeanStokesData();
		float* getAmps();
		unsigned int* getHits();


		PhaseSeries* getPhaseSeries();
		unsigned int getRunningMeanSumLength();
		unsigned int getBinNum();
		unsigned int getNumberOfFreqChans();
		unsigned int getStokesLength();
		unsigned int getCovarianceMatrixLength();
		unsigned int getNumberOfHitChans();
		unsigned int getUnloadCallCount();
		unsigned int getAmpsLength();
		unsigned int getHitsLength();

		void iterateUnloadCallCount();

		bool hasBeenSetup();

	};

}




#endif /* COVARIANCEMATRIXRESULT_H_ */
