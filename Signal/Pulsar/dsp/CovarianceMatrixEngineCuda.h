/*
 * CovarianceMatrixEngineCuda.h
 *
 *  Created on: 01/09/2014
 *      Author: vincentvillani
 */

#ifndef COVARIANCEMATRIXENGINECUDA_H_
#define COVARIANCEMATRIXENGINECUDA_H_

#include "dsp/CovarianceMatrixCuda.h"
#include <cuda_runtime.h>

void computeCovarianceMatrixCUDA(float* d_resultVector, float* d_vector, unsigned int vectorLength,
		unsigned int blockDim2D = 16);

void computeMeanCUDA(float* amps, int ampsLength, float* hits);



#endif /* COVARIANCEMATRIXENGINECUDA_H_ */
