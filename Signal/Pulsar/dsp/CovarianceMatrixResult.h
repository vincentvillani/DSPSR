/*
 * CovarianceMatrixResult.h
 *
 *  Created on: 10/09/2014
 *      Author: vincentvillani
 */

#ifndef COVARIANCEMATRIXRESULT_H_
#define COVARIANCEMATRIXRESULT_H_

namespace dsp
{

	class CovarianceMatrixResult
	{

	private:
		unsigned int _binNum; //number of bins
		unsigned int _freqChanNum; //number of frequency channels
		unsigned int _stokesLength; //Length of the stokes vector
		unsigned int _covarianceMatrixLength; //Array length of a covariance matrix
		unsigned int _hitChanNum; //Number of hit channels
		unsigned int _unloadCalledNum; //Number of times unload has been called


		//pointer to the upper triangular data that constitutes the covariance matrix
		//For each freq channel
		float* _covarianceMatrices;
		float* _runningMeanSum;
		float* _tempMeanStokesData;


	public:
		CovarianceMatrixResult();
		~CovarianceMatrixResult();

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
