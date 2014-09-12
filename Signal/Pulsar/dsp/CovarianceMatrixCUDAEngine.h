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
	//Cuda Kernels
	void outerProductKernel(float* resultMatrix, float* vec, int vectorLength);
	void meanStokesKernel(float* d_amps, unsigned int ampsLength, unsigned int* d_hits, unsigned int stokesLength);
	void applyScale(float* amps, unsigned int ampsLength, double scaleFactor);
	void genericAdd(uint64_t n, float* original, const float* add);
	void genericAdd(unsigned int n, unsigned int* original, const unsigned int* add);


	void computeCovarianceMatrixCUDAEngine(float* d_resultVector, unsigned int resultElementOffset,
			const float* h_amps, float* d_amps, unsigned int ampsLength,
			 const unsigned int* h_hits, unsigned int* d_hits, unsigned int hitsLength,
			 unsigned int stokesLength, double scaleFactor, unsigned int blockDim2D = 16);
};








#endif /* COVARIANCEMATRIXENGINECUDA_H_ */
