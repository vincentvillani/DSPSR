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

class CovarianceMatrixCUDAEngine
{


public:

	CovarianceMatrixCUDAEngine();
	~CovarianceMatrixCUDAEngine();

	void computeCovarianceMatrixCUDAEngine(float* d_resultVector, unsigned int resultElementOffset,
		const float* h_amps, float* d_amps, unsigned int ampsLength,
		 const unsigned int* h_hits, unsigned int* d_hits, unsigned int hitsLength,
		 unsigned int stokesLength, unsigned int blockDim2D = 16);


	void compute_final_covariance_matrices_device(
			float* d_outerProducts, unsigned int outerProductsLength,
			float* d_runningMeanSum, unsigned int runningMeanSumLength,
			unsigned int unloadCalledCount, unsigned int freqChanNum,
			unsigned int covarianceLength, unsigned int ampsLength);



private:

	bool* d_zeroes; //Are zeroes present?
	bool* h_zeroes;

	float* h_tempOuterProducts;
	float* h_tempPhaseOuterProducts;

	float* compute_outer_product_phase_series_device(float* d_runningMeanSum, unsigned int runningMeanSumLength,
			unsigned int unloadCalledCount, unsigned int freqChanNum, unsigned int covarianceLength,
			unsigned int ampsLength);

	bool hitsContainsZeroes(unsigned int* d_hits, unsigned int hitLength);


};



//Cuda Kernels
__global__ void outerProductKernel(float* result, float* vec, int vectorLength);
__global__ void meanStokesKernel(float* d_amps, unsigned int ampsLength, unsigned int* d_hits, unsigned int stokesLength);
__global__ void applyScaleKernel(float* amps, unsigned int ampsLength, double scaleFactor);
__global__ void genericAddKernel(unsigned int n, float* original, const float* add);
__global__ void genericAddKernel(unsigned int n, unsigned int* original, const unsigned int* add);
__global__ void genericDivideKernel(unsigned int n, float* d_numerators, unsigned int denominator);
__global__ void checkForZeroesKernel(float* d_hits, unsigned int hitsLength, bool* d_zeroes);




#endif /* COVARIANCEMATRIXENGINECUDA_H_ */
