/*
 * TimeSeriesCombinerCUDA.h
 *
 *  Created on: 21/09/2014
 *      Author: vincentvillani
 */

#ifndef TIMESERIESCOMBINERCUDA_H_
#define TIMESERIESCOMBINERCUDA_H_

#include <stdio.h>
#include "dsp/TimeSeries.h"
#include "dsp/CovarianceMatrixKernels.h"
#include <cuda_runtime.h>

namespace dsp
{
	class TimeSeriesCombinerCUDA
	{

	public:
		TimeSeriesCombinerCUDA();
		~TimeSeriesCombinerCUDA();

		void combine(TimeSeries* lhs, const TimeSeries* rhs);
	};
}

/*
__global__ void genericAddKernel(unsigned int n, float* original, const float* add);
__global__ void genericAddKernel(unsigned int n, unsigned int* original, const unsigned int* add);
*/


#endif /* TIMESERIESCOMBINERCUDA_H_ */