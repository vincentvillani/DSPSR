/*
 * CovarianceMatrixResult.C
 *
 *  Created on: 10/09/2014
 *      Author: vincentvillani
 */

#include "dsp/CovarianceMatrixResult.h"

	dsp::CovarianceMatrixResult::CovarianceMatrixResult()
	{
		_binNum = 0;
		_freqChanNum = 0;
		_stokesLength = 0;
		_covarianceMatrixLength = 0;
		_hitChanNum = 0;
		_unloadCalledNum = 0;

		_runningMeanSumLength = 0;
		_tempNormalisedAmps = 0;

		_runningMeanSum = NULL;
		_tempNormalisedAmps = NULL;

		d_outerProducts = NULL;
		_outerProductsLength = 0;

		d_amps = NULL;
		d_hits = NULL;



		_useCUDA = false;
		_setup = false;


		//TODO: VINCENT: REMOVE THIS
#if HAVE_CUDA
		_useCUDA = true;
#endif
	}



	dsp::CovarianceMatrixResult::~CovarianceMatrixResult()
	{
		if(_useCUDA)
		{
#if HAVE_CUDA
			cudaFree(_runningMeanSum);
			cudaFree(_tempNormalisedAmps);
			cudaFree(d_outerProducts);
#endif
		}
		else
		{
			if(_runningMeanSum != NULL) //Free running mean sum memory
				delete[] _runningMeanSum;

			if(_tempNormalisedAmps != NULL) //Free temp mean stokes data
				delete[] _tempNormalisedAmps;

			_runningMeanSum = NULL;
			_tempNormalisedAmps = NULL;
		}

	}


	//TODO: VINCENT AT THE MOMENT, IF CUDA IS AVAILABLE, USE IT
	void dsp::CovarianceMatrixResult::setup(unsigned int binNum, unsigned int freqChanNum, unsigned int stokesLength,
			unsigned int covarianceMatrixLength, unsigned int hitChannelNumber)
	{
		_setup = true;

		_binNum = binNum;
		_freqChanNum = freqChanNum;
		_stokesLength = stokesLength;
		_covarianceMatrixLength = covarianceMatrixLength;
		_hitChanNum = hitChannelNumber;
		_unloadCalledNum = 0;

		_runningMeanSumLength = freqChanNum * binNum * stokesLength;
		//_tempMeanStokesDataLength = binNum * stokesLength;
		_outerProductsLength = freqChanNum * covarianceMatrixLength;


		//TODO: VINCENT: FIX THIS
		if(_useCUDA)
		{
#if HAVE_CUDA



			set_ndim(1);
			set_nchan(freqChanNum);
			set_npol(1);
			set_nbit(32);

			//TODO: VINCENT, NO NEED FOR THIS IN THE FINAL VERSION
			cudaMalloc(&d_outerProducts, sizeof(float) * _outerProductsLength);
			cudaMemset(d_outerProducts, 0, sizeof(float) * _outerProductsLength);

			//TODO: VINCENT, THIS WILL AREADY BE ON THE GPU IN THE FINAL VERSION
			//Allocate memory
			//resize(covarianceMatrixLength); //for each freq chan, allocate _covarianceMatrixLength number of 32 bit elements
			//memset(get_data(), 0,  sizeof(float) * covarianceMatrixLength * freqChanNum); //Set to 0


			cudaMalloc(&_runningMeanSum, sizeof(float) * _runningMeanSumLength);
			cudaMalloc(&_tempNormalisedAmps, sizeof(float) * covarianceMatrixLength);

			cudaMemset(_tempNormalisedAmps, 0, sizeof(float) * covarianceMatrixLength);

#endif

		}
		else
		{

			set_ndim(1);
			set_nchan(freqChanNum);
			set_npol(1);
			set_nbit(32);

			//Allocate memory
			resize(covarianceMatrixLength); //for each freq chan, allocate _covarianceMatrixLength number of 32 bit elements
			memset(get_data(), 0,  sizeof(float) * _outerProductsLength); //Set to 0

			_runningMeanSum = new float[_runningMeanSumLength];
			_tempNormalisedAmps = new float[_covarianceMatrixLength];
		}


	}



	float* dsp::CovarianceMatrixResult::getCovarianceMatrix(unsigned int channelOffset)
	{
		return ( (float*)(get_data()) + (channelOffset * _covarianceMatrixLength));
	}



	float* dsp::CovarianceMatrixResult::getRunningMeanSum(unsigned int channelOffset)
	{
		return _runningMeanSum + (channelOffset * _binNum * _stokesLength);
	}



	float* dsp::CovarianceMatrixResult::getTempMeanStokesData()
	{
		return _tempNormalisedAmps;
	}



	unsigned int dsp::CovarianceMatrixResult::getBinNum()
	{
		return _binNum;
	}



	unsigned int dsp::CovarianceMatrixResult::getNumberOfFreqChans()
	{
		return _freqChanNum;
	}



	unsigned int dsp::CovarianceMatrixResult::getStokesLength()
	{
		return _stokesLength;
	}




	unsigned int dsp::CovarianceMatrixResult::getCovarianceMatrixLength()
	{
		return _covarianceMatrixLength;
	}




	unsigned int dsp::CovarianceMatrixResult::getNumberOfHitChans()
	{
		return _hitChanNum;
	}




	unsigned int dsp::CovarianceMatrixResult::getUnloadCallCount()
	{
		return _unloadCalledNum;
	}



	void dsp::CovarianceMatrixResult::iterateUnloadCallCount()
	{
		++_unloadCalledNum;
	}



	bool dsp::CovarianceMatrixResult::hasBeenSetup()
	{
		return _setup;
	}


	float* dsp::CovarianceMatrixResult::getAmps()
	{
		return d_amps;
	}



	unsigned int* dsp::CovarianceMatrixResult::getHits()
	{
		return d_hits;
	}


	unsigned int dsp::CovarianceMatrixResult::getAmpsLength()
	{
		return _binNum * _stokesLength;
	}


	unsigned int dsp::CovarianceMatrixResult::getHitsLength()
	{
		return _binNum;
	}


