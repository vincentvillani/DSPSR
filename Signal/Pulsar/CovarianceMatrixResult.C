/*
 * CovarianceMatrixResult.C
 *
 *  Created on: 10/09/2014
 *      Author: vincentvillani
 */

#include "dsp/CovarianceMatrixResult.h"

	dsp::CovarianceMatrixResult::CovarianceMatrixResult()
	{
		_binNum = NULL;
		_freqChanNum = NULL;
		_stokesLength = NULL;
		_covarianceMatrixLength = NULL;
		_hitChanNum = NULL;
		_unloadCalledNum = NULL;

		_runningMeanSum = NULL;
		_tempMeanStokesData = NULL;

		d_outerProducts = NULL;

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
			cudaFree(_tempMeanStokesData);
			cudaFree(_binNum);
			cudaFree(_freqChanNum);
			cudaFree(_stokesLength);
			cudaFree(_covarianceMatrixLength);
			cudaFree(_unloadCalledNum);
			cudaFree(_runningMeanSum);
			cudaFree(_tempMeanStokesData);
			cudaFree(d_outerProducts);
#endif
		}
		else
		{
			if(_runningMeanSum != NULL) //Free running mean sum memory
				delete[] _runningMeanSum;

			if(_tempMeanStokesData != NULL) //Free temp mean stokes data
				delete[] _tempMeanStokesData;

			if(_binNum != NULL)
				delete _binNum;

			if(_freqChanNum != NULL)
				delete _freqChanNum;

			if(_stokesLength != NULL)
				delete _stokesLength;

			if(_covarianceMatrixLength != NULL)
				delete _covarianceMatrixLength;

			if(_hitChanNum != NULL)
				delete _hitChanNum;

			if(_unloadCalledNum != NULL)
				delete _unloadCalledNum;

			_runningMeanSum = NULL;
			_tempMeanStokesData = NULL;
		}

	}


	//TODO: VINCENT AT THE MOMENT, IF CUDA IS AVAILABLE, USE IT
	void dsp::CovarianceMatrixResult::setup(unsigned int binNum, unsigned int freqChanNum, unsigned int stokesLength,
			unsigned int covarianceMatrixLength, unsigned int hitChannelNumber)
	{
		_setup = true;


		//TODO: VINCENT: FIX THIS
		if(_useCUDA)
		{
#if HAVE_CUDA
			size_t uSize = sizeof(unsigned int);

			cudaMalloc(&_binNum, uSize);
			cudaMalloc(&_freqChanNum, uSize);
			cudaMalloc(&_stokesLength, uSize);
			cudaMalloc(&_covarianceMatrixLength, uSize);
			cudaMalloc(&_hitChanNum, uSize);
			cudaMalloc(&_unloadCalledNum, uSize);

			cudaMemcpy(_binNum, binNum, uSize, cudaMemcpyHostToDevice);
			cudaMemcpy(_freqChanNum, freqChanNum, uSize, cudaMemcpyHostToDevice);
			cudaMemcpy(_stokesLength, stokesLength, uSize, cudaMemcpyHostToDevice);
			cudaMemcpy(_covarianceMatrixLength, covarianceMatrixLength, uSize, cudaMemcpyHostToDevice);
			cudaMemcpy(_hitChanNum, hitChanNum, uSize, cudaMemcpyHostToDevice);
			cudaMemset(_unloadCalledNum, 0, uSize);

			set_ndim(1);
			set_nchan(freqChanNum);
			set_npol(1);
			set_nbit(32);

			//TODO: VINCENT, NO NEED FOR THIS IN THE FINAL VERSION
			cudaMalloc(&d_outerProducts, sizeof(float) * freqChanNum * covarianceMatrixLength);
			cudaMemset(d_outerProducts, 0, sizeof(float) * freqChanNum * covarianceMatrixLength);

			//TODO: VINCENT, THIS WILL AREADY BE ON THE GPU IN THE FINAL VERSION
			//Allocate memory
			resize(covarianceMatrixLength); //for each freq chan, allocate _covarianceMatrixLength number of 32 bit elements
			memset(get_data(), 0,  sizeof(float) * covarianceMatrixLength * freqChanNum); //Set to 0

			cudaMalloc(&_runningMeanSum, sizeof(float) * freqChanNum * binNum * stokesLength);
			cudaMalloc(&_tempMeanStokesData, sizeof(float) * binNum * stokesLength);

			cudaMemset(_tempMeanStokesData, 0, sizeof(float) * freqChanNum * binNum * stokesLength);

#endif

		}
		else
		{
			//Allocate memory
			_binNum = new unsigned int;
			_freqChanNum = new unsigned int;
			_stokesLength = new unsigned int;
			_covarianceMatrixLength = new unsigned int;
			_hitChanNum = new unsigned int;
			_unloadCalledNum = new unsigned int;

			//Set everything first
			*_binNum = binNum;
			*_freqChanNum = freqChanNum;
			*_stokesLength = stokesLength;
			*_covarianceMatrixLength = covarianceMatrixLength;
			*_hitChanNum = hitChannelNumber;
			*_unloadCalledNum = 0;

			set_ndim(1);
			set_nchan(freqChanNum);
			set_npol(1);
			set_nbit(32);

			//Allocate memory
			resize(covarianceMatrixLength); //for each freq chan, allocate _covarianceMatrixLength number of 32 bit elements
			memset(get_data(), 0,  sizeof(float) * covarianceMatrixLength * freqChanNum); //Set to 0

			_runningMeanSum = new float[freqChanNum * binNum * stokesLength];
			_tempMeanStokesData = new float[binNum * stokesLength];
		}


	}



	float* dsp::CovarianceMatrixResult::getCovarianceMatrix(unsigned int channelOffset)
	{
		return ( (float*)(get_data()) + (channelOffset * *_covarianceMatrixLength));
	}



	float* dsp::CovarianceMatrixResult::getRunningMeanSum(unsigned int channelOffset)
	{
		return _runningMeanSum + (channelOffset * *_binNum * *_stokesLength);
	}



	float* dsp::CovarianceMatrixResult::getTempMeanStokesData()
	{
		return _tempMeanStokesData;
	}



	unsigned int dsp::CovarianceMatrixResult::getBinNum()
	{
		return *_binNum;
	}



	unsigned int dsp::CovarianceMatrixResult::getNumberOfFreqChans()
	{
		return *_freqChanNum;
	}



	unsigned int dsp::CovarianceMatrixResult::getStokesLength()
	{
		return *_stokesLength;
	}




	unsigned int dsp::CovarianceMatrixResult::getCovarianceMatrixLength()
	{
		return *_covarianceMatrixLength;
	}




	unsigned int dsp::CovarianceMatrixResult::getNumberOfHitChans()
	{
		return *_hitChanNum;
	}




	unsigned int dsp::CovarianceMatrixResult::getUnloadCallCount()
	{
		return *_unloadCalledNum;
	}



	void dsp::CovarianceMatrixResult::iterateUnloadCallCount()
	{
		++(*_unloadCalledNum);
	}



	bool dsp::CovarianceMatrixResult::hasBeenSetup()
	{
		return _setup;
	}
