/*
 * CovarianceMatrixKernels.h
 *
 *  Created on: 23/09/2014
 *      Author: vincentvillani
 */

#ifndef COVARIANCEMATRIXKERNELS_H_
#define COVARIANCEMATRIXKERNELS_H_

#include <cuda_runtime.h>

//Cuda Kernels
__global__ void outerProductKernel(float* result, unsigned int resultLength, float* vec, unsigned int vecLength);
__global__ void meanStokesKernel(float* d_amps, unsigned int ampsLength, const unsigned int* d_hits, unsigned int stokesLength);
__global__ void applyScaleKernel(float* amps, unsigned int ampsLength, double scaleFactor);
__global__ void genericAddKernel(unsigned int n, float* original, const float* add);
__global__ void genericAddKernel(unsigned int n, unsigned int* original, const unsigned int* add);
__global__ void genericSubtractionKernel(unsigned int n, float* original, const float* sub);
__global__ void genericDivideKernel(unsigned int n, float* d_numerators, unsigned int denominator);
__global__ void checkForZeroesKernel(const unsigned int* d_hits, unsigned int hitsLength, bool* d_zeroes);


#endif /* COVARIANCEMATRIXKERNELS_H_ */
