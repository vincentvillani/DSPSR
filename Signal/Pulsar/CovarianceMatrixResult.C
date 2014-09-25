/*
 * CovarianceMatrixResult.C
 *
 *  Created on: 10/09/2014
 *      Author: vincentvillani
 */

#include "dsp/CovarianceMatrixResult.h"


//TODO: DEBUG

#if HAVE_CUDA
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#endif

	dsp::CovarianceMatrixResult::CovarianceMatrixResult()
	{
		_binNum = 0;
		_freqChanNum = 0;
		_stokesLength = 0;
		_covarianceMatrixLength = 0;
		_hitChanNum = 0;
		_unloadCalledNum = 0;

		_phaseSeries = NULL;

		_runningMeanSumLength = 0;

		_runningMeanSum = NULL;

		d_outerProducts = NULL;
		_outerProductsLength = 0;

		_amps = NULL;
		d_hits = NULL;

		_useCUDA = false;
		_setup = false;



		//TODO: VINCENT: REMOVE THIS
#if HAVE_CUDA
		_useCUDA = true;
		printf("USE CUDA IS TRUE!\n");
#endif
	}



	dsp::CovarianceMatrixResult::~CovarianceMatrixResult()
	{
		if(_useCUDA)
		{
#if HAVE_CUDA
			printf("CUDA DESTRUCTOR RUN!!!!!!!!!!!!!!!!!!\n");

			cudaFree(_runningMeanSum);
			cudaFree(d_outerProducts);
			cudaFree(_amps);
			cudaFree(d_hits);
#endif
		}
		else
		{
			if(_runningMeanSum != NULL) //Free running mean sum memory
				delete[] _runningMeanSum;

			if(_amps != NULL)
				delete[] _amps;

			_runningMeanSum = NULL;
		}

		delete _phaseSeries;

	}


	//TODO: VINCENT AT THE MOMENT, IF CUDA IS AVAILABLE, USE IT
	void dsp::CovarianceMatrixResult::setup(unsigned int binNum, unsigned int freqChanNum, unsigned int stokesLength,
			unsigned int covarianceMatrixLength, unsigned int hitChannelNumber, const PhaseSeries* ps)
	{
		_setup = true;

		//Clone the initial phaseSeries
		_phaseSeries = new PhaseSeries();
		_phaseSeries->set_memory( ps->memory ); //TODO: VINCENT: IS THIS CORRECT?
		_phaseSeries->set_hits_memory( ps->hits_memory ); //TODO: VINCENT: IS THIS CORRECT?
		_phaseSeries->operator =(*ps);

		_binNum = binNum;
		_freqChanNum = freqChanNum;
		_stokesLength = stokesLength;
		_covarianceMatrixLength = covarianceMatrixLength;
		_hitChanNum = hitChannelNumber;
		_unloadCalledNum = 0;

		_runningMeanSumLength = freqChanNum * binNum * stokesLength;
		_outerProductsLength = freqChanNum * covarianceMatrixLength;


		//TODO: VINCENT: FIX THIS
		if(_useCUDA)
		{
#if HAVE_CUDA

			printf("CUDA SETUP RUN!\n");
			set_ndim(1);
			set_nchan(freqChanNum);
			set_npol(1);
			set_nbit(32);


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
		}


	}



	dsp::PhaseSeries*  dsp::CovarianceMatrixResult::getPhaseSeries()
	{
		return _phaseSeries;
	}


	float* dsp::CovarianceMatrixResult::getCovarianceMatrix(unsigned int channelOffset)
	{
		if(!_useCUDA)
			return ( (float*)(get_data()) + (channelOffset * _covarianceMatrixLength));
		else
			return d_outerProducts + (channelOffset * _covarianceMatrixLength);
	}




	float* dsp::CovarianceMatrixResult::getRunningMeanSum(unsigned int channelOffset)
	{
		return _runningMeanSum + (channelOffset * _binNum * _stokesLength);
	}


	unsigned int dsp::CovarianceMatrixResult::getRunningMeanSumLength()
	{
		return _freqChanNum * _binNum * _stokesLength;
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


	unsigned int dsp::CovarianceMatrixResult::getAmpsLength()
	{
		return _binNum * _stokesLength;
	}


	unsigned int dsp::CovarianceMatrixResult::getHitsLength()
	{
		return _binNum;
	}


