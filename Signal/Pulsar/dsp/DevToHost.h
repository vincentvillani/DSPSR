/*
 * DevToHost.h
 *
 *  Created on: 25/09/2014
 *      Author: vincentvillani
 */

#ifndef DEVTOHOST_H_
#define DEVTOHOST_H_

#include "dsp/PhaseSeries.h"

class PhaseSeries;

namespace dsp
{
	//Abstract base class for an object that transfers data from a device (Most likely a GPU) to a host (CPU)
	//Device -> host
	class DevToHost
	{
	public:

		DevToHost(){};
		virtual ~DevToHost(){};
		virtual void transfer(const PhaseSeries* from, PhaseSeries* to) = 0;
	};

}


#endif /* DEVTOHOST_H_ */
