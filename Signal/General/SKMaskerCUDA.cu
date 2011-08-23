//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2010 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SKMaskerCUDA.h"

#include <iostream>

using namespace std;

CUDA::SKMaskerEngine::SKMaskerEngine (cudaStream_t _stream)
{
  stream = _stream;
}

void CUDA::SKMaskerEngine::setup (unsigned _nchan, unsigned _npol, unsigned _span)
{
  if (dsp::Operation::verbose)
    cerr << "CUDA::SKMaskerEngine::setup nchan=" << _nchan << " npol=" << _npol
         << " span=" << _span << endl;

  nchan = _nchan;
  npol = _npol;
  span = _span;
}


/* cuda kernel to mask 1 channel for both polarisations */
__global__ void mask1chan (unsigned char * mask_base,
           float * out_base,
           unsigned npol,
           unsigned end,
           unsigned span)
{
  // ichan = blockIdx.x * blockDim.x + threadIdx.x

  float * p0 = out_base + span * npol * (blockIdx.x * blockDim.x + threadIdx.x);
  float * p1 = out_base + span * npol * (blockIdx.x * blockDim.x + threadIdx.x) + span;

  mask_base += (blockIdx.x * blockDim.x + threadIdx.x);

  if (mask_base[0])
  {
    for (unsigned j=0; j<end; j++)
    {
      p0[j] = 0;
      p1[j] = 0;
    }
  }

}

void CUDA::SKMaskerEngine::perform (dsp::BitSeries* mask, unsigned mask_offset, 
           dsp::TimeSeries * output, unsigned offset, unsigned end)
{

  if (dsp::Operation::verbose)
    cerr << "CUDA::SKMaskerEngine::perform mask_offset=" << mask_offset << " offset=" << offset << " end=" << end << endl;

  // order is FPT
  float * out_base = output->get_datptr(0, 0) + offset;
  unsigned char * mask_base = mask->get_datptr() + mask_offset;

  dim3 threads (128);
  dim3 blocks (nchan/threads.x);
 
  mask1chan<<<blocks,threads,0,stream>>> (mask_base, out_base, npol, end, span);

  if (dsp::Operation::record_time)
  {
    cudaThreadSynchronize ();

    cudaError error = cudaGetLastError();
    if (error != cudaSuccess)
      throw Error (InvalidState, "CUDA::SKMaskerEngine::perform",
       cudaGetErrorString (error));
  }

}
