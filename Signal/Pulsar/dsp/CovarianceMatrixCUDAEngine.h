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
#include "dsp/PhaseSeries.h"
#include "dsp/CovarianceMatrixResult.h"

namespace dsp
{

class CovarianceMatrixCUDAEngine
{


public:

	CovarianceMatrixCUDAEngine();
	~CovarianceMatrixCUDAEngine();

	void computeCovarianceMatricesCUDA(const PhaseSeries* ps, CovarianceMatrixResult* covarianceMatrixResult);





	float* compute_final_covariance_matrices_device(CovarianceMatrixResult* cmr);



private:


	bool* d_zeroes; //Are zeroes present?
	bool h_zeroes;

	//Compute a covariance matrix for one freq channel
	void computeCovarianceMatrix(float* d_result,
			const float* h_amps, float* d_amps, unsigned int ampsLength,
			unsigned int* d_hits, unsigned int hitsLength,
			float* d_runningMean,
			unsigned int stokesLength, unsigned int blockDim2D = 16);


	float* compute_outer_product_phase_series_device(CovarianceMatrixResult* cmr);

	bool hitsContainsZeroes(unsigned int* d_hits, unsigned int hitLength);
	const unsigned int* getHitsPtr(const PhaseSeries* phaseSeriesData, CovarianceMatrixResult* covarianceMatrixResult, int freqChan);

};

}


//Cuda Kernels
__global__ void outerProductKernel(float* result, float* vec, unsigned int vectorLength);
__global__ void meanStokesKernel(float* d_amps, unsigned int ampsLength, unsigned int* d_hits, unsigned int stokesLength);
__global__ void applyScaleKernel(float* amps, unsigned int ampsLength, double scaleFactor);
__global__ void genericAddKernel(unsigned int n, float* original, const float* add);
__global__ void genericAddKernel(unsigned int n, unsigned int* original, const unsigned int* add);
__global__ void genericSubtractionKernel(unsigned int n, float* original, const float* sub);
__global__ void genericDivideKernel(unsigned int n, float* d_numerators, unsigned int denominator);
__global__ void checkForZeroesKernel(unsigned int* d_hits, unsigned int hitsLength, bool* d_zeroes);




#endif /* COVARIANCEMATRIXENGINECUDA_H_ */
