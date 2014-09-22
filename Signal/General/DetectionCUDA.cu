//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/DetectionCUDA.h"

#include "Error.h"
#include "cross_detect.h"
#include "stokes_detect.h"
#include "templates.h"
#include "debug.h"

#include <memory>

#include <string.h>

using namespace std;

void check_error (const char*);

/*
  PP   = p^* p
  QQ   = q^* q
  RePQ = Re[p^* q]
  ImPQ = Im[p^* q]
*/

#define COHERENCE(PP,QQ,RePQ,ImPQ,p,q) \
  PP   = (p.x * p.x) + (p.y * p.y); \
  QQ   = (q.x * q.x) + (q.y * q.y); \
  RePQ = (p.x * q.x) + (p.y * q.y); \
  ImPQ = (p.x * q.y) - (p.y * q.x);

#define STOKES(I,Q,U,V,p,q) \
  I = (p.x * p.x) + (p.y * p.y) + (q.x * q.x) + (q.y * q.y); \
  Q = (p.x * p.x) + (p.y * p.y) - (q.x * q.x) - (q.y * q.y); \
  U = 2 * ((p.x * q.x) + (p.y * q.y)); \
  V = 2 * ((p.x * q.y) - (p.y * q.x));

#ifdef __INTERLEAVED_VOLTAGES

/*
 * The following code assumes that the voltage samples from each polarization are
 * interleaved; e.g. t0p0re t0p0im t0p1re t0p1im t1p0re ...
 */

__global__ void coherence4 (float4* base, uint64_t span)
{
  base += blockIdx.y * span;
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

  float2 p, q;
  float4 result = base[i];

  p.x = result.w;
  p.y = result.x;
  q.x = result.y;
  q.y = result.z;

  COHERENCE4 (result,p,q);

  base[i] = result;
}

/*
  The input data are arrays of ndat pairs of complex numbers: 

  Re[p0],Im[p0],Re[p1],Im[p1]

  There are nchan such arrays; base pointers are separated by span.
*/

void polarimetry_ndim4 (float* data, uint64_t span,
			uint64_t ndat, unsigned nchan)
{
  int threads = 256;

  dim3 blocks;
  blocks.x = ndat/threads;
  blocks.y = nchan;

  coherence4<<<blocks,threads>>> ((float4*)data, span/4); 

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error ("CUDA::DetectionEngine::polarimetry_ndim4");
}

#endif

/*
  The input data are pairs of arrays of ndat complex numbers: 

  Re[p0],Im[p0] ...

  Re[p1],Im[p1] ...

  There are nchan such pairs of arrays; base pointers p0 and p1 are 
  separated by span.
*/

#define COHERENCE2(s0,s1,p,q) COHERENCE(s0.x,s0.y,s1.x,s1.y,p,q)

// for each channel index (blockIdx.y) skip two voltage polarizations
#define NPOL_PER_CHANNEL 2

__global__ void coherence2 (float2* base, unsigned span, unsigned ndat)
{
  float2* p0 = base + blockIdx.y * span * NPOL_PER_CHANNEL;
  float2* p1 = p0 + span;

  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= ndat)
    return;

  float2 s0, s1;

  COHERENCE2(s0,s1,p0[i],p1[i]);

  p0[i] = s0;
  p1[i] = s1;
}

#define COHERENCE4(r,p,q) COHERENCE(r.w,r.x,r.y,r.z,p,q)
#define STOKES4(r,p,q) STOKES(r.w,r.x,r.y,r.z,p,q)

__global__ void coherence4 (bool stokes, float2* in_base, unsigned in_span, unsigned ndat, float* out_base, unsigned out_span)
{
  float2* p0 = in_base + blockIdx.y * in_span * NPOL_PER_CHANNEL;
  float2* p1 = p0 + in_span;

  float4* s = (float4*)(out_base + blockIdx.y * out_span);

  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= ndat)
    return;

  if (stokes)
  {
	  STOKES4(s[i],p0[i],p1[i]);
  }
  else
  {
	  COHERENCE4(s[i],p0[i],p1[i]);
  }
}

CUDA::DetectionEngine::DetectionEngine (cudaStream_t _stream)
{
  stream = _stream;
}

void CUDA::DetectionEngine::polarimetry (unsigned ndim,
					 const dsp::TimeSeries* input, 
					 dsp::TimeSeries* output)
{
	if (ndim != 2 && ndim != 4)
		throw Error (InvalidParam, "CUDA::DetectionEngine::polarimetry",
				"implemented only for ndim==2 and ndim==4 (ndim=%u)", ndim);

	bool inplace = input == output;
	if (!inplace && ndim == 2)
		throw Error (InvalidParam, "CUDA::DetectionEngine::polarimetry"
				"cannot handle out-of-place data when ndim==2");

	if (inplace && ndim == 4)
		throw Error (InvalidParam, "CUDA::DetectionEngine::polarimetry"
				"can only handle out-of-place data when ndim==4");

  uint64_t ndat = output->get_ndat ();
  unsigned nchan = output->get_nchan ();

  unsigned ichan=0, ipol=0;

  float* out_base = output->get_datptr (ichan=0, ipol=0);
  uint64_t out_span = output->get_datptr (ichan=0, ipol=1) - out_base;

  float* in_base = output->get_datptr (ichan=0, ipol=0);
  uint64_t in_span = output->get_datptr (ichan=0, ipol=1) - in_base;

  if (dsp::Operation::verbose)
    cerr << "CUDA::DetectionEngine::polarimetry ndim=" << output->get_ndim () 
         << " ndat=" << ndat << " span=" << span << endl;

  dim3 threads (128);
  dim3 blocks (ndat/threads.x, nchan);

  if (ndat % threads.x)
    blocks.x ++;

  // pass span as number of complex values
  if (ndim == 2)
	  coherence2<<<blocks,threads,0,stream>>> ((float2*)in_base, in_span/2, ndat);
  else if (ndim == 4)
	  coherence4<<<blocks.threads,0,stream>>> (state == Signal::Stokes, (float2*)in_base, in_span/2, ndat, out_base, out_span);

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error ("CUDA::DetectionEngine::polarimetry");
}

