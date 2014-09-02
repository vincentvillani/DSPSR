/*
 * CovarianceMatrixEngineCuda.h
 *
 *  Created on: 01/09/2014
 *      Author: vincentvillani
 */

#ifndef COVARIANCEMATRIXENGINECUDA_H_
#define COVARIANCEMATRIXENGINECUDA_H_

#include "CovarianceMatrixCUDA.h"


void computeCovarianceMatrixCUDA(float* d_resultVector, unsigned int resultByteOffset, float* amps, unsigned int ampsLength,
		float* hits, unsigned int hitsLength, unsigned int blockDim2D = 16);




#endif /* COVARIANCEMATRIXENGINECUDA_H_ */
