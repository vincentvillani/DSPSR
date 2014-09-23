/*
 * PhaseSeriesCombinerCUDA.C
 *
 *  Created on: 21/09/2014
 *      Author: vincentvillani
 */

#include "dsp/PhaseSeriesCombinerCUDA.h"
#include "dsp/PhaseSeries.h"
#include "dsp/CovarianceMatrixKernels.h"


dsp::PhaseSeriesCombinerCUDA::PhaseSeriesCombinerCUDA()
{
	_tsc = new dsp::TimeSeriesCombinerCUDA();
}


dsp::PhaseSeriesCombinerCUDA::~PhaseSeriesCombinerCUDA()
{
	delete _tsc;
}




void dsp::PhaseSeriesCombinerCUDA::combine(PhaseSeries* const lhs, const PhaseSeries* rhs)
{
	if(lhs == NULL || rhs == NULL)
		return;

	if(rhs->get_nbin() == 0 || rhs->get_integration_length() == 0.0)
		return;

	if( !lhs->mixable(*rhs, rhs->get_nbin() ) )
		return;


	//TODO: VINCENT: ADD THIS PART BACK IN
	//combine the time series part
	_tsc->combine(lhs, rhs);

	const unsigned int hitLength = lhs->get_nbin() * lhs->hits_nchan;
	unsigned int nHitChan = lhs->get_hits_nchan();

	unsigned int* d_lhsHits = lhs->hits;
	unsigned int* d_rhsHits = rhs->hits;

	unsigned int blockDim = 256;
	unsigned int gridDim = min ( (unsigned int)ceil(hitLength / blockDim), 65535);

	for(unsigned int i = 0; i < nHitChan; ++i)
	{
		printf("Launching GenericAddKernel with Grid Dim: %u, Block Dim: %u\n", gridDim, blockDim);
		genericAddKernel <<<gridDim, blockDim>>> (hitLength, d_lhsHits + (i * hitLength), d_rhsHits + (i * hitLength));
	}

	lhs->integration_length += rhs->integration_length;
	lhs->ndat_total += rhs->ndat_total;

	if (!lhs->ndat_expected)
		lhs->ndat_expected = rhs->ndat_expected;

}







