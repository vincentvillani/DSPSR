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
#include <string>
#include <sstream>

#include "CovarianceMatrixResult.h"

#if HAVE_CUDA
#include "dsp/CovarianceMatrixCUDAEngine.h"
#endif


namespace dsp
{
	class CovarianceMatrix : public PhaseSeriesUnloader
	{

	protected:

		//unloader that will write the data to disk, when the time comes
		Reference::To<PhaseSeriesUnloader> _unloader;

		//----- result data  ------
		PhaseSeries* _phaseSeries;
		CovarianceMatrixResult* _covarianceMatrixResult;



/*
#if HAVE_CUDA

		//Device specific variables / functions
		float* _d_resultVector;
		//float* _d_vector;
		float* _d_amps;
		unsigned int* _d_hits;

		void setup_device(unsigned int chanNum, unsigned int hitChanNum, unsigned int binNum, unsigned int nPol, unsigned int nDim); //allocate memory if we are using a device/cuda
		void compute_covariance_matrix_device(const PhaseSeries* phaseSeriesData);

#endif
*/

		//void setup_host(unsigned int chanNum, unsigned int hitChanNum, unsigned int binNum, unsigned int nPol, unsigned int nDim); //allocate memory if we are using the host
		//void compute_covariance_matrix_host(const PhaseSeries* phaseSeriesData);
		//void scale_and_mean_stokes_data_host(const float* stokesData, const unsigned int* hits, unsigned int chan);
		//void covariance_matrix_host(unsigned int freqChan);
		//void compute_final_covariance_matrices_host();
		//float** compute_outer_product_phase_series_host_old();
		//float** compute_outer_product_phase_series_host_new();

		//Both cuda and normal methods
		float* convertToSymmetric(float* upperTriangle, unsigned int rowLength);
		void printSymmetricMatrix(float* symmetricMatrix, int rowLength, bool genFile);
		void outputSymmetricMatrix(float* symmetricMatrix, unsigned int rowLength, std::string filename);
		void printUpperTriangularMatrix(float* result, int rowLength, bool genFile);
		void outputUpperTriangularMatrix(float* result, unsigned int rowLength, std::string filename);
		void copyAndPrint(float* deviceData, int arrayLength, int rowLength);

		unsigned int covariance_matrix_length(const unsigned int numBin);

		//const unsigned int* getHitsPtr(const PhaseSeries* phaseSeriesData, int freqChan);


	public:

		//constructors/destructors
		CovarianceMatrix();
		PhaseSeriesUnloader* clone () const{return NULL;} //TODO: VINCENT: ACTUALLY IMPLEMENT THIS
		virtual ~CovarianceMatrix();

		void unload(const PhaseSeries*);
		void set_minimum_integration_length (double seconds){}; //TODO: VINCENT: ACTUALLY IMPLEMENT THIS //if integration length is less than the minimum, discard it
		void set_unloader(PhaseSeriesUnloader* unloader);
	};
}




#endif


