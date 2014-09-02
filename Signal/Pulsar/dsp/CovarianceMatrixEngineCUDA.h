/*
 * CovarianceMatrixEngineCuda.h
 *
 *  Created on: 01/09/2014
 *      Author: vincentvillani
 */

#ifndef COVARIANCEMATRIXENGINECUDA_H_
#define COVARIANCEMATRIXENGINECUDA_H_

#include "CovarianceMatrixCUDA.h"
#include <stdio.h>


void computeCovarianceMatrixCUDAEngine(float* d_resultVector, unsigned int resultByteOffset,
		const float* h_amps, float* d_amps, unsigned int ampsLength,
		 const unsigned int* h_hits, float* d_hits, unsigned int hitsLength,
		 unsigned int stokesLength, double scaleFactor, unsigned int blockDim2D = 16);




#endif /* COVARIANCEMATRIXENGINECUDA_H_ */
