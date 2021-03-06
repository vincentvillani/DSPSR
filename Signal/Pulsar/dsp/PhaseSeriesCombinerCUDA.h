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

		//TODO: VINCENT: REMOVE THESE IN THE FINAL VERSION
		unsigned int* d_temp_data1;
		unsigned int* d_temp_data2;

	public:

		PhaseSeriesCombinerCUDA();
		~PhaseSeriesCombinerCUDA();

		void combine(PhaseSeries* lhs, const PhaseSeries* rhs);
	};
}



#endif /* PHASESERIESCOMBINERCUDA_H_ */
