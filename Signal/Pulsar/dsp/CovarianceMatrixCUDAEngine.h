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
	__host__ void computeCovarianceMatrixCUDAEngine(float* d_resultVector, unsigned int resultElementOffset,
			const float* h_amps, float* d_amps, unsigned int ampsLength,
			 const unsigned int* h_hits, unsigned int* d_hits, unsigned int hitsLength,
			 unsigned int stokesLength, double scaleFactor, unsigned int blockDim2D = 16);
};



//Cuda Kernels
__global__ void outerProductKernel(float* resultMatrix, float* vec, int vectorLength);
__global__ void meanStokesKernel(float* d_amps, unsigned int ampsLength, unsigned int* d_hits, unsigned int stokesLength);
__global__ void applyScale(float* amps, unsigned int ampsLength, double scaleFactor);
__global__ void genericAdd(unsigned int n, float* original, const float* add);
__global__ void genericAdd(unsigned int n, unsigned int* original, const unsigned int* add);






#endif /* COVARIANCEMATRIXENGINECUDA_H_ */
