/*
 * OuterProduct.h
 *
 *  Created on: 27/08/2014
 *      Author: vincentvillani
 */


#ifndef COVARIANCEMATRIX_H
#define COVARIANCEMATRIX_H

#include "PhaseSeriesUnloader.h"
#include "PhaseSeries.h"

namespace dsp
{
	class CovarianceMatrix : public PhaseSeriesUnloader
	{

	protected:

		//result data
		float* _covarianceMatrix;
		float* _ampsData;
		unsigned int* _hitsData;

		//reusable data
		float* _meanStokesData; //mean stokes data for this iteration


		void computeCovarianceMatrix();


	public:

		//constructors/destructors
		CovarianceMatrix();
		PhaseSeriesUnloader* clone () const{return NULL;}; //TODO: VINCENT: ACTUALLY IMPLEMENT THIS
		virtual ~CovarianceMatrix() {};

		void unload(const PhaseSeries*);
		void set_minimum_integration_length (double seconds){};




		//! Handle partially completed PhaseSeries data
		//virtual void partial (const dsp::PhaseSeries*) {};

	};
}




#endif


