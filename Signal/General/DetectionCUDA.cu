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

__global__ void coherence4 (bool stokes, const float2* in_base, unsigned in_span, unsigned ndat, float4* out_base, unsigned out_span)
{
  const float2* p0 = in_base + blockIdx.y * in_span * NPOL_PER_CHANNEL;
  const float2* p1 = p0 + in_span;

  float4* s = out_base + blockIdx.y * out_span;

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

void CUDA::DetectionEngine::polarimetry (Signal::State state, unsigned ndim,
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

  if (dsp::Operation::verbose)
    cerr << "CUDA::DetectionEngine::polarimetry ndim=" << output->get_ndim () 
         << " npol=" << output->get_npol() << " ndat=" << ndat << endl;

  dim3 threads (128);
  dim3 blocks (ndat/threads.x, nchan);

  if (ndat % threads.x)
    blocks.x ++;

  // pass span as number of complex values
  if (ndim == 2)
  {
    float* out_base = output->get_datptr (0,0);
    uint64_t out_span = output->get_nfloat_span();

    // convert out_span in floats to span in float2 by dividing by 2
    coherence2<<<blocks,threads,0,stream>>> ((float2*)out_base, out_span/2, ndat);

  }
  else if (ndim == 4)
  {
    float* out_base = output->get_datptr (0,0);
    uint64_t out_span = output->get_nfloat_span ();

cerr << "computing input span" << endl;
    const float* in_base = input->get_datptr (0,0);
    uint64_t in_span = input->get_nfloat_span ();
cerr << "launching coherence4 out_base=" << out_base << " in_base=" << in_base << endl;

    // convert in_span in floats to span in float2 by dividing by 2
    // convert out_span in floats to span in float4 by dividing by 4
    coherence4<<<blocks,threads,0,stream>>> (state == Signal::Stokes, (const float2*)in_base, in_span/2, ndat, (float4*) out_base, out_span/4);

cerr << "coherence4 ok" << endl;
}

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error ("CUDA::DetectionEngine::polarimetry");
}

