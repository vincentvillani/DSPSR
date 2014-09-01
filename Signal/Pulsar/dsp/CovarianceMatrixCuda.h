/*
 * CovarianceMatrixCuda.h
 *
 *  Created on: 01/09/2014
 *      Author: vincentvillani
 */

#ifndef COVARIANCEMATRIXCUDA_H_
#define COVARIANCEMATRIXCUDA_H_



__global__ void outerProductKernel(float* resultMatrix, float* vec, int vectorLength);

__global__ void meanStokesKernel(float* amps, unsigned int ampsLength, float* hits);

#endif /* COVARIANCEMATRIXCUDA_H_ */
