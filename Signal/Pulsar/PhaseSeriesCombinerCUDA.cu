/*
 * PhaseSeriesCombinerCUDA.C
 *
 *  Created on: 21/09/2014
 *      Author: vincentvillani
 */

#include "dsp/PhaseSeriesCombinerCUDA.h"
#include "dsp/PhaseSeries.h"



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


	//combine the time series part
	//_tsc->combine(lhs, rhs);

	const unsigned int nhits = lhs->get_nbin() * lhs->hits_nchan;

	unsigned int* h_lhsHits = lhs->hits;
	unsigned int* h_rhsHits = lhs->hits;
	unsigned int* d_lhsHits;
	unsigned int* d_rhsHits;

	//TODO: VINCENT, THIS WILL ALREADY BE ON THE DEVICE IN THE FINAL VERSION, NO NEED FOR COPIES
	cudaMalloc(&d_lhsHits, sizeof(unsigned int) * nhits);
	cudaMalloc(&d_rhsHits, sizeof(unsigned int) * nhits);

	cudaMemcpy(d_lhsHits, h_lhsHits, sizeof(unsigned int) * nhits, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rhsHits, h_rhsHits, sizeof(unsigned int) * nhits, cudaMemcpyHostToDevice);

	unsigned int blockDim = 256;
	unsigned int gridDim = min ( (unsigned int)ceil(nhits / gridDim), 65535);

	printf("Launching GenericAddKernel with Grid Dim: %u, Block Dim: %u\n", gridDim, blockDim);
	genericAddKernel <<<gridDim, blockDim>>> (nhits, d_lhsHits, d_rhsHits);

	cudaMemcpy(h_lhsHits, d_lhsHits, sizeof(unsigned int) * nhits, cudaMemcpyDeviceToHost);

	lhs->integration_length += rhs->integration_length;
	lhs->ndat_total += rhs->ndat_total;

	if (!lhs->ndat_expected)
		lhs->ndat_expected = rhs->ndat_expected;

	cudaFree(d_lhsHits);
	cudaFree(d_rhsHits);

/*
	  if (verbose)
		cerr << "dsp::PhaseSeries::combine"
				" this=" << this << " that=" << prof << endl;

	  if (!prof || prof->get_nbin() == 0)
		return;

	  if (verbose)
		cerr << "dsp::PhaseSeries::combine length add="
			 << prof->integration_length
			 << " current=" << integration_length << endl;

	  if (!integration_length)
	  {
		if (verbose)
		  cerr << "dsp::PhaseSeries::combine this is empty" << endl;

		*this = *prof;
		return;
	  }

	  if (!mixable (*prof, prof->get_nbin()))
		throw Error (InvalidParam, "PhaseSeries::combine",
			 "PhaseSeries !mixable");

	  TimeSeries::operator += (*prof);

	  const unsigned nhits = get_nbin() * hits_nchan;
	  for (unsigned ihit=0; ihit<nhits; ihit++)
		hits[ihit] += prof->hits[ihit];

	  integration_length += prof->integration_length;
	  ndat_total += prof->ndat_total;

	  if (!ndat_expected)
		ndat_expected = prof->ndat_expected;
	}
	 catch (Error& error)
	   {
	     throw error += "dsp::PhaseSeries::combine";
	   }
*/
}







