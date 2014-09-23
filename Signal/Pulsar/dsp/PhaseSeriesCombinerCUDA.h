/*
 * PhaseSeriesCombinerCUDA.h
 *
 *  Created on: 21/09/2014
 *      Author: vincentvillani
 */

#ifndef PHASESERIESCOMBINERCUDA_H_
#define PHASESERIESCOMBINERCUDA_H_

#include "ReferenceAble.h"
#include "dsp/TimeSeriesCombinerCUDA.h"




namespace dsp
{
	class PhaseSeries;

	class PhaseSeriesCombinerCUDA : public Reference::Able
	{
	private:
		TimeSeriesCombinerCUDA* _tsc;

	public:

		PhaseSeriesCombinerCUDA();
		~PhaseSeriesCombinerCUDA();

		void combine(PhaseSeries* const lhs, const PhaseSeries* rhs);


	};
}



#endif /* PHASESERIESCOMBINERCUDA_H_ */
