/*
 * CovarianceMatrixResult.h
 *
 *  Created on: 10/09/2014
 *      Author: vincentvillani
 */

#ifndef COVARIANCEMATRIXRESULT_H_
#define COVARIANCEMATRIXRESULT_H_

#include <iostream>
#include "dsp/DataSeries.h"


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

		//TODO: VINCENT: DO THIS PROPERLY - REMOVE THIS
		bool _useCUDA;

		float* _runningMeanSum; //Running total of the mean for each freq channel
		float* _tempMeanStokesData; //scratch space for one covariance matrix length


	public:
		CovarianceMatrixResult();
		~CovarianceMatrixResult();

		void setup(unsigned int binNum, unsigned int freqChanNum, unsigned int stokesLength,
					unsigned int covarianceMatrixLength, unsigned int hitChannelNumber);
		void setup(); //Allocate memory, once everything has been set


		//Getters and Setters
		float* getCovarianceMatrices();
		float* getRunningMeanSum();
		float* getTempMeanStokesData();

		unsigned int getBinNum();
		void setBinNum(unsigned int binNum);

		unsigned int getNumberOfFreqChans();
		void setNumberOfFreqChans(unsigned int freqChan);

		unsigned int getStokesLength();
		void setStokesLength(unsigned int stokesLength);

		unsigned int getCovarianceMatrixLength();
		void setCovarianceMatrixLength(unsigned int covLength);

		unsigned int getNumberOfHitChans();
		void setNumberOfHitChans(unsigned int hitChanNum);

		unsigned int getUnloadCallCount();
		void incrementUnloadCallCount();

	};

}




#endif /* COVARIANCEMATRIXRESULT_H_ */
