/*
 * DevToHostCuda.cu
 *
 *  Created on: 25/09/2014
 *      Author: vincentvillani
 */

#include "dsp/DevToHostCuda.h"


dsp::DevToHostCuda::DevToHostCuda(cudaStream_t stream)
{
	_tpsc = new TransferPhaseSeriesCUDA(stream);
}

dsp::DevToHostCuda::~DevToHostCuda()
{
	//DO NOTHING?
}


void dsp::DevToHostCuda::transfer(const PhaseSeries* from, PhaseSeries* to)
{
	_tpsc->set_kind( cudaMemcpyDeviceToHost );
	_tpsc->set_input( from );
	_tpsc->set_output( to );
	_tpsc->set_transfer_hits( true );
	_tpsc->operate ();
}



