//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/Detection.h,v $
   $Revision: 1.3 $
   $Date: 2002/10/11 02:31:13 $
   $Author: wvanstra $ */


#ifndef __Detection_h
#define __Detection_h

#include "Operation.h"
#include "Observation.h"

namespace dsp {

  //! Detects phase-coherent floating-point Timeseries data
  /*!  This class may be used to perform simple square law detection
    or calculation of the Signal::Stokes parameters or the coherency matrix.  
    In the case of Signal::Stokes/Coherency, the components may be formed in
    polarization-major order or time-major order. 
  */
  class Detection : public Operation {

  public:
    
    //! Constructor
    Detection ();
    
    //! Destructor
    ~Detection () { }

    //! Set the state of the output data
    void set_output_state (Signal::State _state);
    //! Get the state of the output data
    Signal::State get_output_state () const { return state; }

    //! Set the dimension of the output data
    void set_output_ndim (int _ndim) { ndim = _ndim; }
    //! Get the dimension of the output data
    bool get_output_ndim () const { return ndim; }

  protected:

    //! Detect the input data
    virtual void operation ();

    //! Signal::State of the output data
    Signal::State state;

    //! Dimension of the output data
    int ndim;

    //! Perform simple square-law detection
    void square_law ();

    //! Perform simple square-law detection
    void polarimetry ();

    //! Set the state of the output Timeseries
    void resize_output ();
  };

}

#endif // !defined(__Detection_h)
