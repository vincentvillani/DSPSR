//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/AutoCorrelation.h,v $
   $Revision: 1.1 $
   $Date: 2004/11/08 13:57:20 $
   $Author: wvanstra $ */

#ifndef __AutoCorrelation_h
#define __AutoCorrelation_h

#include "dsp/Convolution.h"

namespace dsp {
  
  //! Forms lag spectra in any number of frequency channels and polarizations
  class AutoCorrelation: public Transformation <TimeSeries, TimeSeries> {

  public:

    //! Null constructor
    AutoCorrelation ();

    //! Set the number of lags
    void set_nlag (unsigned _nlag) { nlag = _nlag; }

    //! Get the number of lags
    unsigned get_nlag () const { return nlag; } 

  protected:

    //! Perform the convolution transformation on the input TimeSeries
    virtual void transformation ();

    //! Number of lags
    unsigned nlag;

  };
  
}

#endif