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
		_binNum = binNum;
		_freqChanNum = freqChanNum;
		_stokesLength = stokesLength;
		_covarianceMatrixLength = covarianceMatrixLength;
		_hitChanNum = hitChannelNumber;

		set_ndim(1);
		set_nchan(_freqChanNum);
		set_npol(1);
		set_nbit(32);
		resize(_covarianceMatrixLength); //for each freq chan, allocate _covarianceMatrixLength number of 32 bit elements

		if(_useCUDA)
		{

		}
		else
		{
			_runningMeanSum = new float[_freqChanNum * _binNum * _covarianceMatrixLength];
			_tempMeanStokesData = new float[_covarianceMatrixLength];
		}

	}


	void dsp::CovarianceMatrixResult::setup()
	{
		set_ndim(1);
		set_nchan(_freqChanNum);
		set_npol(1);
		set_nbit(32);
		resize(_covarianceMatrixLength); //for each freq chan, allocate _covarianceMatrixLength number of 32 bit elements

		if(_useCUDA)
		{

		}
		else
		{
			_runningMeanSum = new float[_freqChanNum * _binNum * _covarianceMatrixLength];
			_tempMeanStokesData = new float[_covarianceMatrixLength];
		}
	}



	float* dsp::CovarianceMatrixResult::getRunningMeanSum()
	{
		return _runningMeanSum;
	}



	float* dsp::CovarianceMatrixResult::getTempMeanStokesData()
	{
		return _tempMeanStokesData;
	}



	unsigned int dsp::CovarianceMatrixResult::getBinNum()
	{
		return _binNum;
	}



	void dsp::CovarianceMatrixResult::setBinNum(unsigned int binNum)
	{
		_binNum = binNum;
	}



	unsigned int dsp::CovarianceMatrixResult::getNumberOfFreqChans()
	{
		return _freqChanNum;
	}



	void dsp::CovarianceMatrixResult::setNumberOfFreqChans(unsigned int freqChan)
	{
		_freqChanNum = freqChan;
	}



	unsigned int dsp::CovarianceMatrixResult::getStokesLength()
	{
		return _stokesLength;
	}



	void dsp::CovarianceMatrixResult::setStokesLength(unsigned int stokesLength)
	{
		_stokesLength = stokesLength;
	}



	unsigned int dsp::CovarianceMatrixResult::getCovarianceMatrixLength()
	{
		return _covarianceMatrixLength;
	}



	void dsp::CovarianceMatrixResult::setCovarianceMatrixLength(unsigned int covLength)
	{
		_covarianceMatrixLength = covLength;
	}



	unsigned int dsp::CovarianceMatrixResult::getNumberOfHitChans()
	{
		return _hitChanNum;
	}



	void dsp::CovarianceMatrixResult::setNumberOfHitChans(unsigned int hitChanNum)
	{
		_hitChanNum = hitChanNum;
	}



	unsigned int dsp::CovarianceMatrixResult::getUnloadCallCount()
	{
		return _unloadCalledNum;
	}



	void dsp::CovarianceMatrixResult::incrementUnloadCallCount()
	{
		++_unloadCalledNum;
	}
