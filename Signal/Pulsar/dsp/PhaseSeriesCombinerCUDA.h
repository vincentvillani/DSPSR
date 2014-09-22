/*
 * PhaseSeriesCombinerCUDA.h
 *
 *  Created on: 21/09/2014
 *      Author: vincentvillani
 */

#ifndef PHASESERIESCOMBINERCUDA_H_
#define PHASESERIESCOMBINERCUDA_H_

#include "dsp/PhaseSeries.h"
#include "Kernel/Classes/dsp/TimeSeriesCombinerCUDA.h"


class PhaseSeries;

namespace dsp
{
	class PhaseSeriesCombinerCUDA
	{
	private:
		TimeSeriesCombinerCUDA* _tsc;

	public:

		PhaseSeriesCombinerCUDA();
		~PhaseSeriesCombinerCUDA();

		void combine(PhaseSeries* const lhs, const PhaseSeries* rhs);


	};
}


__global__ void genericAddKernel(unsigned int n, unsigned int* original, const unsigned int* add);


#endif /* PHASESERIESCOMBINERCUDA_H_ */
