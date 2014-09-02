/*
 * CovarianceMatrixCuda.h
 *
 *  Created on: 01/09/2014
 *      Author: vincentvillani
 */

#ifndef COVARIANCEMATRIXCUDA_H_
#define COVARIANCEMATRIXCUDA_H_

#include <cuda_runtime.h>
#include <stdio.h>


__global__ void outerProductKernel(float* resultMatrix, float* vec, int vectorLength);

__global__ void meanStokesKernel(float* d_amps, unsigned int ampsLength, unsigned int* d_hits, unsigned int stokesLength);


__global__ void applyScale(float* amps, unsigned int ampsLength, double scaleFactor);

#endif /* COVARIANCEMATRIXCUDA_H_ */
