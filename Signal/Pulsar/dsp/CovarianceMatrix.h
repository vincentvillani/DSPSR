/*
 * OuterProduct.h
 *
 *  Created on: 27/08/2014
 *      Author: vincentvillani
 */


#ifndef COVARIANCEMATRIX_H
#define COVARIANCEMATRIX_H

//Include the 'HAVE_CUDA' macro, if it is defined
#if HAVE_CONFIG_H
#include <config.h>
#endif

#include <cstring>
#include <iostream>
#include <string>
#include <sstream>

#include "CovarianceMatrixResult.h"
#include "PhaseSeriesUnloader.h"
#include "PhaseSeries.h"

//Include the CUDA engine for computing the covariance matrices
#if HAVE_CUDA
#include "CovarianceMatrixCUDAEngine.h"
#endif

//Forward dec of the engine that computes the covariance matrices, is done just incase
//'CovarianceMatrixCUDAEngine.h' is not included
class CovarianceMatrixCUDAEngine;

namespace dsp
{
	class CovarianceMatrix : public PhaseSeriesUnloader
	{

	protected:

		//unloader that will write the data to disk, when the time comes
		Reference::To<PhaseSeriesUnloader> _unloader;

		//Reference::To<Memory> _memory;

		//Holds the preliminary results of the covariance matrix as it is being iteratively calculated
		CovarianceMatrixResult* _covarianceMatrixResult;

		//An engine that calculates the covariance matrix on the GPU
		CovarianceMatrixCUDAEngine* _engine;

		//calculates an outer product of an integrated PS and then combines the passed phase series with the passed phase series passed to unload()
		void compute_covariance_matrix_host(const PhaseSeries* phaseSeriesData);

		//Helper method to get a pointer to the hit array of a Phase Series for a given channel
		const unsigned int* getHitsPtr(const PhaseSeries* phaseSeriesData, int freqChan);

		//Normalises the amps data (i.e. the stokesData) by dividing it by the hits value
		//It also adds the normalised amps to a running total, which will eventually be used to calculate
		//the covariance matrix
		void norm_stokes_data_host(const float* stokesData, const unsigned int* hits, unsigned int chan);


		//used in the compute_final_covariance_matrices_host() method as a helper function to compute
		//the outer product of the running mean that has been accumulating every iteration
		//(i.e. every time unload() was called)
		float* compute_outer_product_phase_series_host();

		//Uses compute_outer_product_phase_series_host() to compute the running mean's outer product
		//then subtracts that from the outer products calculated during calls to unload()
		void compute_final_covariance_matrices_host();

		//calculates the number of non zero components in a upper triangluar matrix
		//if numBin = rowLength
		//(rowLength * (rowLength + 1)) / 2
		unsigned int covariance_matrix_length(const unsigned int numBin);

		//has a cuda engine been set?
		bool engine_set();



		//  ---------- Debug Helper functions ----------

		//Turns a upper triangular matrix into a symmetrical (along the diagonal) 'full' matrix
		float* convertToSymmetric(float* upperTriangle, unsigned int rowLength);

		//Prints a 'full symmetric' matrix to the console
		void printSymmetricMatrix(float* symmetricMatrix, int rowLength, bool genFile);

		//Prints an upper triangular matrix to the console
		void printUpperTriangularMatrix(float* result, int rowLength, bool genFile);

		//Writes a 'full symmetric' matrix to a file
		void outputSymmetricMatrix(float* symmetricMatrix, unsigned int rowLength, std::string filename);

		//Writes an upper triangluar matrix to a file
		void outputUpperTriangularMatrix(float* result, unsigned int rowLength, std::string filename);

		//Helper function to copy and print values from a cuda device
		void copyAndPrint(float* deviceData, int arrayLength, int rowLength);


		bool _useCuda; //Should cuda be used?


	public:

		CovarianceMatrix(bool useCuda);
		//TODO: VINCENT: ACTUALLY IMPLEMENT THIS
		PhaseSeriesUnloader* clone () const{return NULL;}

		//At the moment, the destructor of the covariance matrix is what does the final calculations to get
		//the covariance matrix
		virtual ~CovarianceMatrix();

		//unload should be called when a phase series is to be written to a file
		//instead, the CovarianceMatrix class takes it and calculates outer products
		//to eventually compute the covariance matrix
		void unload(const PhaseSeries* ps);

		//TODO: VINCENT: ACTUALLY IMPLEMENT THIS //if integration length is less than the minimum, discard it
		void set_minimum_integration_length (double seconds){};

		//Sets the unloader,
		//the unloader should be used to actually write the file to disk when the covariance matrix
		//is done with everything
		void set_unloader(PhaseSeriesUnloader* unloader);

		//Sets the cuda engine
		void set_engine(CovarianceMatrixCUDAEngine* engine);


	};
}




#endif


