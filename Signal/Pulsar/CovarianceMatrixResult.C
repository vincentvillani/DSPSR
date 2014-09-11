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

		_runningMeanSum = NULL;
		_tempMeanStokesData = NULL;

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

		}
		else
		{
			if(_runningMeanSum != NULL) //Free running mean sum memory
				delete[] _runningMeanSum;

			if(_tempMeanStokesData != NULL) //Free temp mean stokes data
				delete[] _tempMeanStokesData;
		}

	}



	void dsp::CovarianceMatrixResult::setup(unsigned int binNum, unsigned int freqChanNum, unsigned int stokesLength,
			unsigned int covarianceMatrixLength, unsigned int hitChannelNumber)
	{

		//Set everything first
		_binNum = binNum;
		_freqChanNum = freqChanNum;
		_stokesLength = stokesLength;
		_covarianceMatrixLength = covarianceMatrixLength;
		_hitChanNum = hitChannelNumber;
		_setup = true;

		set_ndim(1);
		set_nchan(_freqChanNum);
		set_npol(1);
		set_nbit(32);


		//Allocate memory
		resize(_covarianceMatrixLength); //for each freq chan, allocate _covarianceMatrixLength number of 32 bit elements
		memset(get_data(), 0,  sizeof(float) * _covarianceMatrixLength * _freqChanNum); //Set to 0

		//const Memory* mem = get_memory();

		//_runningMeanSum = mem->allocate(_freqChanNum * _binNum * _covarianceMatrixLength * sizeof(float));
		//_tempMeanStokesData = mem->allocate(_covarianceMatrixLength * sizeof(float));


		if(_useCUDA)
		{

		}
		else
		{
			_runningMeanSum = new float[_freqChanNum * _binNum * _stokesLength];
			_tempMeanStokesData = new float[_binNum * _stokesLength];

		}


	}



	float* dsp::CovarianceMatrixResult::getCovarianceMatrix(unsigned int channelOffset)
	{
		return ( (float*)(get_data()) + (channelOffset * _covarianceMatrixLength));
	}



	float* dsp::CovarianceMatrixResult::getRunningMeanSum(unsigned int channelOffset)
	{
		return _runningMeanSum + (channelOffset * _covarianceMatrixLength);
	}



	float* dsp::CovarianceMatrixResult::getTempMeanStokesData()
	{
		return _tempMeanStokesData;
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



	bool dsp::CovarianceMatrixResult::hasBeenSetup()
	{
		return _setup;
	}
