/*
 * OuterProduct.h
 *
 *  Created on: 27/08/2014
 *      Author: vincentvillani
 */


#ifndef COVARIANCEMATRIX_H
#define COVARIANCEMATRIX_H

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "PhaseSeriesUnloader.h"
#include "PhaseSeries.h"
#include <cstring>
#include <iostream>

//#if HAVE_CUDA

#include "dsp/CovarianceMatrixEngineCUDA.h"

//#endif


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


		//helper functions

		void compute_covariance_matrix_host(unsigned int freqChan);
		unsigned int covariance_matrix_length(const unsigned int numBin);


//#ifndef HAVE_CUDA
		//Host specific variables / functions

		float* _tempMeanStokesData;

		void setup_host(unsigned int chanNum, unsigned int binNum, unsigned int nPol, unsigned int nDim); //allocate memory if we are using the host
		void compute_covariance_matrix_host(const PhaseSeries* phaseSeriesData);
		void scale_and_mean_stokes_data_host(const float* stokesData, const unsigned int* hits, double scale);


//#else
		//Device specific variables / functions
		float* _d_resultVector;
		//float* _d_vector;
		float* _d_amps;
		unsigned int* _d_hits;

		void setup_device(unsigned int chanNum, unsigned int binNum, unsigned int nPol, unsigned int nDim); //allocate memory if we are using a device/cuda
		void compute_covariance_matrix_device(const PhaseSeries* phaseSeriesData);

		void printResultUpperTriangular(float* result, int rowLength, bool genFile);
		void copyAndPrint(float* deviceData, int arrayLength, int rowLength);
//#endif

	public:

		//constructors/destructors
		CovarianceMatrix();
		PhaseSeriesUnloader* clone () const{return NULL;} //TODO: VINCENT: ACTUALLY IMPLEMENT THIS
		virtual ~CovarianceMatrix();

		void unload(const PhaseSeries*);
		void set_minimum_integration_length (double seconds){}; //if integration length is less than the minimum, discard it
		void set_unloader(PhaseSeriesUnloader* unloader);
	};
}




#endif


