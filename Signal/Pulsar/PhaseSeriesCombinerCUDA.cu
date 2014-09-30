/*
 * PhaseSeriesCombinerCUDA.C
 *
 *  Created on: 21/09/2014
 *      Author: vincentvillani
 */

#include "dsp/PhaseSeriesCombinerCUDA.h"
#include "dsp/PhaseSeries.h"
#include "dsp/CovarianceMatrixKernels.h"
#include "dsp/Memory.h"
#include <iostream>


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


dsp::PhaseSeriesCombinerCUDA::PhaseSeriesCombinerCUDA()
{
	_tsc = new dsp::TimeSeriesCombinerCUDA();
	d_temp_data1 = NULL;
	d_temp_data2 = NULL;

}



dsp::PhaseSeriesCombinerCUDA::~PhaseSeriesCombinerCUDA()
{
	delete _tsc;

	if(d_temp_data1 != NULL)
		cudaFree(d_temp_data1);

	if(d_temp_data2 != NULL)
		cudaFree(d_temp_data2);
}



void dsp::PhaseSeriesCombinerCUDA::combine(PhaseSeries* lhs, const PhaseSeries* rhs)
{
	if(lhs == NULL || rhs == NULL)
	{
		fprintf(stderr,"Returning 1\n");
		return;
	}

	//TODO: VINCENT: WHY ISNT INTEGRATION LENGTH COPIED OVER?
	if(rhs->get_nbin() == 0) //|| rhs->get_integration_length() == 0.0)
	{
		fprintf(stderr,"BIN: %u, IL: %f\n", rhs->get_nbin(), rhs->get_integration_length());
		fprintf(stderr,"Returning 2\n");
		return;
	}

	if( !lhs->mixable(*rhs, rhs->get_nbin() ) )
	{
		fprintf(stderr,"Returning 3\n");
		return;
	}


	//TODO: VINCENT: ADD THIS PART BACK IN
	//combine the time series part
	fprintf(stderr,"BEFORE TSC\n");
	_tsc->combine(lhs, rhs);
	fprintf(stderr,"AFTER TSC\n");

	const unsigned int hitLength = rhs->get_nbin() * rhs->hits_nchan;
	unsigned int nHitChan = rhs->get_hits_nchan();
	unsigned int totalHitLength = hitLength * nHitChan;

	unsigned int* h_lhsHits = lhs->hits;
	unsigned int* h_rhsHits = rhs->hits;

	unsigned int blockDim = 256;
	unsigned int gridDim = min ( (unsigned int)ceil(totalHitLength / blockDim), 65535);


	fprintf(stderr,"PS: %p, %p\n", h_lhsHits, h_rhsHits);

	if( rhs->get_memory()->on_host() )
		fprintf(stderr,"RHS MEMORY IS ON HOST\n");
	else
		fprintf(stderr,"RHS MEMORY IS NOT ON HOST\n");

	if( lhs->get_memory()->on_host() )
		fprintf(stderr,"LHS MEMORY IS ON HOST\n");
	else
		fprintf(stderr,"LHS MEMORY IS NOT ON HOST\n");


	fprintf(stderr,"PHASE SERIES COMBINE: Launching GenericAddKernel with Grid Dim: %u, Block Dim: %u\n", gridDim, blockDim);
	genericAddKernel <<<gridDim, blockDim>>> (totalHitLength, h_lhsHits, h_rhsHits);


	//TODO: VINCENT: DEBUG
	cudaError_t error2 = cudaDeviceSynchronize();
	if(error2 != cudaSuccess)
	{
		fprintf(stderr,"CUDA ERROR: %s\n", cudaGetErrorString(error2));
		exit(2);
	}


	lhs->integration_length += rhs->integration_length;
	lhs->ndat_total += rhs->ndat_total;

	if (!lhs->ndat_expected)
		lhs->ndat_expected = rhs->ndat_expected;

}







