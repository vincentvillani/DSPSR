/*
 * OuterProduct.h
 *
 *  Created on: 27/08/2014
 *      Author: vincentvillani
 */


#ifndef COVARIANCEMATRIX_H
#define COVARIANCEMATRIX_H

#include "Archiver.h"

namespace dsp
{
	class CovarianceMatrix : public PhaseSeriesUnloader
	{

	protected:

		void computeCovarianceMatrix();

	public:

		//constructors/destructors
		CovarianceMatrix();
		PhaseSeriesUnloader* clone () const{return NULL;};
		virtual ~CovarianceMatrix() {};

		void unload(const PhaseSeries*){};
		void set_minimum_integration_length (double seconds){};




		//! Handle partially completed PhaseSeries data
		//virtual void partial (const dsp::PhaseSeries*) {};

	};
}




#endif


