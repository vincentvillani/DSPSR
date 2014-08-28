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
#include <vector.h>

namespace dsp
{
	class CovarianceMatrix : public PhaseSeriesUnloader
	{

	protected:

		unsigned int _binNum; //number of bins
		unsigned int _freqChanNum; //number of frequency channels
		unsigned int _stokesLength; //Length of the stokes vector
		unsigned int _covarianceMatrixLength;

		//unloader that will write the data to disk, when the time comes
		PhaseSeriesUnloader* _unloader;



		//----- result data  ------
		//may need to keep a running total of everything, use a phase series and the += operator to do it
		PhaseSeries* _phaseSeries;

		//pointer to the upper triangular data that constitutes the covariance matrix
		//For each freq channel
		float** _covarianceMatrices;

		//temp data that is reused over and over again
		//mean stokes data (i.e. stokes/host), for each freq channel
		//float** _summedMeanStokesDatas; //Data is stored as all the I's across the phase bins, then all Q's across the phase bins etc, for each freq channel

		float* _tempMeanStokesData;


		//helper functions

		void compute_covariance_matrix_host(unsigned int freqChan);
		unsigned int covariance_matrix_length(const unsigned int numBin);

		void mean_stokes_data_host(const float* stokesData, const unsigned int* hits, unsigned int offset);
		//void add_temp_stokes_data_host(); //TODO: VINCENT: FIGURE OUT IF THIS IS THE CORRECT THING TO DO

	public:

		//constructors/destructors
		CovarianceMatrix();
		PhaseSeriesUnloader* clone () const{return NULL;} //TODO: VINCENT: ACTUALLY IMPLEMENT THIS
		virtual ~CovarianceMatrix();

		void unload(const PhaseSeries*);
		void set_minimum_integration_length (double seconds){};
		void set_unloader(PhaseSeriesUnloader* unloader);

	};
}




#endif


