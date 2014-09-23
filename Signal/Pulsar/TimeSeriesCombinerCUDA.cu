/*
 * TimeSeriesCombinerCUDA.C
 *
 *  Created on: 21/09/2014
 *      Author: vincentvillani
 */

#include "dsp/TimeSeriesCombinerCUDA.h"

dsp::TimeSeriesCombinerCUDA::TimeSeriesCombinerCUDA()
{

}


dsp::TimeSeriesCombinerCUDA::~TimeSeriesCombinerCUDA()
{

}


void dsp::TimeSeriesCombinerCUDA::combine(TimeSeries* lhs, const TimeSeries* rhs)
{
	if(lhs == NULL || rhs == NULL)
		return;

	//TODO: ASK WILLEM ABOUT THIS
	/*
	if(lhs->get_ndat() == 0)
	{
		lhs->operator=(rhs);
		return;
	}
	*/

	if(!lhs->combinable(*lhs))
	{
		return;
	}

	if(lhs->get_ndat() != rhs->get_ndat())
	{
		return;
	}

	uint64_t npt = lhs->get_ndat() * lhs->get_ndim();
	unsigned int blockDim = 256;
	unsigned int gridDim;

	float* d_data1;


	if(lhs->get_order() == dsp::TimeSeries::OrderTFP)
	{
		npt *= lhs->get_nchan() * lhs->get_npol();
		gridDim = min ( (unsigned int)ceil(npt / blockDim), 65535);

		d_data1 = lhs->get_dattfp();
		const float* d_data2 = rhs->get_dattfp();

		printf("Launching GenericAddKernel with Grid Dim: %u, Block Dim: %u\n", gridDim, blockDim);
		genericAddKernel <<< gridDim, blockDim >>> (npt, d_data1, d_data2);

		//TODO: VINCENT: DEBUG
		cudaError_t error2 = cudaPeekAtLastError();
		if(error2 != cudaSuccess)
		{
			printf("CUDA ERROR: %s\n", cudaGetErrorString(error2));
			exit(2);
		}

		return;
	}


	gridDim = min ( (unsigned int)ceil(npt / blockDim), 65535);

	for (unsigned ichan = 0; ichan < lhs->get_nchan(); ++ichan)
	{
		for (unsigned ipol = 0; ipol < lhs->get_npol(); ++ipol)
		{
			d_data1 = lhs->get_datptr (ichan, ipol);
			const float* d_data2 = rhs->get_datptr (ichan, ipol);

			printf("TIME SERIES COMBINE: Launching GenericAddKernel with Grid Dim: %u, Block Dim: %u\n", gridDim, blockDim);
			genericAddKernel <<< gridDim, blockDim >>> (npt, d_data1, d_data2);

			//TODO: VINCENT: DEBUG
			cudaError_t error2 = cudaPeekAtLastError();
			if(error2 != cudaSuccess)
			{
				printf("CUDA ERROR: %s\n", cudaGetErrorString(error2));
				exit(2);
			}
		}
	}
}




/*
//Kernel for generically adding things on the GPU
__global__ void genericAddKernel(unsigned int n, float* original, const float* add)
{
	for(unsigned int absIdx = blockDim.x * blockIdx.x + threadIdx.x; absIdx < n; absIdx += gridDim.x * blockDim.x)
	{
		original[absIdx] += add[absIdx];
	}
}


//Kernel for generically adding things on the GPU
__global__ void genericAddKernel(unsigned int n, unsigned int* original, const unsigned int* add)
{
	for(unsigned int absIdx = blockDim.x * blockIdx.x + threadIdx.x; absIdx < n; absIdx += gridDim.x * blockDim.x)
	{
		original[absIdx] += add[absIdx];
	}
}
*/


