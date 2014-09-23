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



void dsp::PhaseSeriesCombinerCUDA::combine(PhaseSeries* const lhs, const PhaseSeries* rhs)
{
	if(lhs == NULL || rhs == NULL)
	{
		//printf("Returning 1\n");
		return;
	}

	if(rhs->get_nbin() == 0 || rhs->get_integration_length() == 0.0)
	{
		//printf("Returning 2\n");
		return;
	}

	if( !lhs->mixable(*rhs, rhs->get_nbin() ) )
	{
		//printf("Returning 3\n");
		return;
	}


	//TODO: VINCENT: ADD THIS PART BACK IN
	//combine the time series part
	_tsc->combine(lhs, rhs);

	const unsigned int hitLength = lhs->get_nbin() * lhs->hits_nchan;
	unsigned int nHitChan = lhs->get_hits_nchan();
	unsigned int totalHitLength = hitLength * nHitChan;

	unsigned int* h_lhsHits = lhs->hits;
	unsigned int* h_rhsHits = rhs->hits;

	unsigned int blockDim = 256;
	unsigned int gridDim = min ( (unsigned int)ceil(totalHitLength / blockDim), 65535);



	if(rhs->get_memory()->on_host())
		printf("RHS MEMORY IS ON HOST\n");
	else
		printf("RHS MEMORY IS NOT ON HOST\n");


	//TODO: VINCENT: NO NEED TO DO THIS IN THE FINAL VERSION
	if(d_temp_data1 == NULL || d_temp_data2 == NULL)
	{
		cudaMalloc(&d_temp_data1, sizeof(unsigned int) * totalHitLength);
		cudaMalloc(&d_temp_data2, sizeof(unsigned int) * totalHitLength);
	}

	//TODO: VINCENT: NO NEED TO DO THIS IN THE FINAL VERSION
	cudaMemcpy(d_temp_data1, h_lhsHits, sizeof(unsigned int) * totalHitLength, cudaMemcpyHostToDevice);
	cudaMemcpy(d_temp_data2, h_rhsHits, sizeof(unsigned int) * totalHitLength, cudaMemcpyHostToDevice);

	printf("PHASE SERIES COMBINE: Launching GenericAddKernel with Grid Dim: %u, Block Dim: %u\n", gridDim, blockDim);
	genericAddKernel <<<gridDim, blockDim>>> (totalHitLength, d_temp_data1, d_temp_data2);

	//TODO: VINCENT: DEBUG
	cudaError_t error2 = cudaPeekAtLastError();
	if(error2 != cudaSuccess)
	{
		printf("CUDA ERROR: %s\n", cudaGetErrorString(error2));
		exit(2);
	}

	//TODO: VINCENT: NO NEED TO DO THIS IN THE FINAL VERSION
	//copy the data back to the host
	cudaMemcpy(h_lhsHits, d_temp_data1, sizeof(unsigned int) * totalHitLength, cudaMemcpyDeviceToHost);

	/*
	for(unsigned int i = 0; i < nHitChan; ++i)
	{
		/*
		if(lhs->get_memory()->on_host())
		{
			printf("MEMORY IS ON HOST\n");
		}
		else
			printf("MEMORY IS NOT ON HOST\n");


		//TODO: VINCENT NO NEED TO COPY IN THE FINAL VERSION?

		printf("PHASE SERIES COMBINE: Launching GenericAddKernel with Grid Dim: %u, Block Dim: %u\n", gridDim, blockDim);
		genericAddKernel <<<gridDim, blockDim>>> (hitLength, d_lhsHits + (i * hitLength), d_rhsHits + (i * hitLength));
	}
*/

	lhs->integration_length += rhs->integration_length;
	lhs->ndat_total += rhs->ndat_total;

	if (!lhs->ndat_expected)
		lhs->ndat_expected = rhs->ndat_expected;

}







