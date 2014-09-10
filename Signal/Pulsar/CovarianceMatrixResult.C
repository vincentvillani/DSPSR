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

		_covarianceMatrices = NULL;
		_runningMeanSum = NULL;
		_tempMeanStokesData = NULL;
	}



	dsp::CovarianceMatrixResult::~CovarianceMatrixResult()
	{

		if(_covarianceMatrices != NULL) //Free covariance matrix memory
		{
			for(int i = 0; i < _freqChanNum; ++i)
				delete[] _covarianceMatrices[i];

			delete[] _covarianceMatrices;
		}


		if(_runningMeanSum != NULL) //Free running mean sum memory
		{
			for(int i = 0; i < _freqChanNum; ++i)
				delete[] _runningMeanSum[i];

			delete[] _runningMeanSum;
		}

		delete[] _tempMeanStokesData; //Free temp mean stokes data

	}



	void dsp::CovarianceMatrixResult::setup()
	{

	}



	float* dsp::CovarianceMatrixResult::getCovarianceMatrices()
	{
		return _covarianceMatrices;
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
