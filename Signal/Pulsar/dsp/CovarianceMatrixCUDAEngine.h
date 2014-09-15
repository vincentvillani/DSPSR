/*
 * CovarianceMatrixEngineCuda.h
 *
 *  Created on: 01/09/2014
 *      Author: vincentvillani
 */

#ifndef COVARIANCEMATRIXCUDAENGINE_H_
#define COVARIANCEMATRIXCUDAENGINE_H_

#include <stdio.h>
#include <cuda_runtime.h>
#include "PhaseSeries.h"
#include "CovarianceMatrixResult.h"

class CovarianceMatrixCUDAEngine
{


public:

	CovarianceMatrixCUDAEngine();
	~CovarianceMatrixCUDAEngine();

	void computeCovarianceMatricesCUDA(const PhaseSeries* ps, CovarianceMatrixResult* covarianceMatrixResult);




	/*
	void compute_final_covariance_matrices_device(
			float* d_outerProducts, unsigned int outerProductsLength,
			float* d_runningMeanSum, unsigned int runningMeanSumLength,
			unsigned int unloadCalledCount, unsigned int freqChanNum,
			unsigned int covarianceLength, unsigned int ampsLength);
			*/



private:

	bool* d_zeroes; //Are zeroes present?
	bool h_zeroes;

	//float* d_amps; //scratch space for amps and hits on the device
	//float* d_hits;

	float* h_tempOuterProducts;
	float* h_tempPhaseOuterProducts;

	//Compute a covariance matrix for one freq channel
	void computeCovarianceMatrix(float* d_resultVector,
		const float* h_amps, float* d_amps, unsigned int ampsLength,
		 const unsigned int* h_hits, unsigned int* d_hits, unsigned int hitsLength,
		 unsigned int stokesLength, unsigned int blockDim2D = 16);


	float* compute_outer_product_phase_series_device(float* d_runningMeanSum, unsigned int runningMeanSumLength,
			unsigned int unloadCalledCount, unsigned int freqChanNum, unsigned int covarianceLength,
			unsigned int ampsLength);

	bool hitsContainsZeroes(unsigned int* d_hits, unsigned int hitLength);


};








#endif /* COVARIANCEMATRIXENGINECUDA_H_ */
