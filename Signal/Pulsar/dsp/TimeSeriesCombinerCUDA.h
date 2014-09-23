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
#include "dsp/Memory.h"
#include <cuda_runtime.h>

namespace dsp
{
	class TimeSeriesCombinerCUDA : public Reference::Able
	{

	public:
		TimeSeriesCombinerCUDA();
		~TimeSeriesCombinerCUDA();

		void combine(TimeSeries* lhs, const TimeSeries* rhs);
	};
}

#endif /* TIMESERIESCOMBINERCUDA_H_ */
