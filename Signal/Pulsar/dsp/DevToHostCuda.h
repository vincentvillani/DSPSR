/*
 * DevToHostCuda.h
 *
 *  Created on: 25/09/2014
 *      Author: vincentvillani
 */

#ifndef DEVTOHOSTCUDA_H_
#define DEVTOHOSTCUDA_H_

#include "dsp/DevToHost.h"
#include "dsp/TransferPhaseSeriesCUDA.h"

namespace dsp
{
	class DevToHostCuda : public DevToHost
	{
	private:
		Reference::To<TransferPhaseSeriesCUDA> _tpsc;

	public:
		DevToHostCuda(cudaStream_t stream);
		~DevToHostCuda();

		void transfer(const PhaseSeries* from, PhaseSeries* to);
	};
}


#endif /* DEVTOHOSTCUDA_H_ */
