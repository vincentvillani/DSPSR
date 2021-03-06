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
#include "dsp/CovarianceMatrixKernels.h"
#include "Kernel/Classes/dsp/MemoryCUDA.h"

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

	CUDA::DeviceMemory* h_deviceMemory;

	bool* d_zeroes; //Are zeroes present?
	bool h_zeroes;

	float* d_ampsScratch; //Scratch space for all amps values in a freq channel
	unsigned int h_ampsScratchLength; //Length of ampsScratch i.e. stokesLength * nbin


	void computeCovarianceMatrix(CovarianceMatrixResult* cmr, const PhaseSeries* ps);
	float* compute_outer_product_phase_series_device(CovarianceMatrixResult* cmr); //Compute the outer product for a phase series
	bool hitsContainsZeroes(const unsigned int* d_hits, unsigned int hitLength); //Does this array contain any zeroes?

	const unsigned int* getHitsPtr(const PhaseSeries* phaseSeriesData, CovarianceMatrixResult* covarianceMatrixResult, int freqChan);
	void outputUpperTriangularMatrix(float* result, unsigned int rowLength, std::string filename);

};

}

#endif /* COVARIANCEMATRIXENGINECUDA_H_ */
