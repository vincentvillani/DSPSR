/*
 * CovarianceMatrix.C
 *
 *  Created on: 27/08/2014
 *      Author: vincentvillani
 */


#include "dsp/CovarianceMatrix.h"




dsp::CovarianceMatrix::CovarianceMatrix()
{
	_stokesLength = 4; //TODO: VINCENT - MAKE THIS THIS VARIABLE SOMEHOW


	//initially set pointers to null
	_phaseSeries = NULL;
	_covarianceMatrices = NULL;
	_tempMeanStokesData = NULL;
	_unloader = NULL;
}


dsp::CovarianceMatrix::~CovarianceMatrix()
{
	delete _phaseSeries; //TODO: VINCENT: IS THIS CORRECT?
	delete _unloader; //TODO: VINCENT: IS THIS CORRECT?
	delete [] _tempMeanStokesData;

	for(int i = 0; i < _freqChanNum; ++i)
		delete [] _covarianceMatrices[i];

	delete [] _covarianceMatrices;

}



void dsp::CovarianceMatrix::unload(const PhaseSeries* phaseSeriesData)
{

	unsigned int binNum = phaseSeriesData->get_nbin();

	std::cerr << "dsp::CovarianceMatrix::unload Freq Chans: " << phaseSeriesData->get_hits_nchan() << ", Binsize: "
			<< binNum << ", NPol: " << phaseSeriesData->get_npol() << ", NDim: " << phaseSeriesData->get_ndim()
			<< std::endl;


	//first time this is called numBins may be zero
	if(binNum == 0)
		return;

	//Allocate memory on first use, once we know the amount of memory required
	if(_covarianceMatrices == NULL) //&& binNum != 0)
	{

		#if !(HAVE_CUDA)
			setup_host ( phaseSeriesData->get_nchan(), binNum, phaseSeriesData->get_npol(), phaseSeriesData->get_ndim() );
		#else
			setup_device ( phaseSeriesData->get_nchan(), binNum, phaseSeriesData->get_npol(), phaseSeriesData->get_ndim() );

		#endif

	}


	#if !(HAVE_CUDA)
		compute_covariance_matrix_host(phaseSeriesData);
	#else
		compute_covariance_matrix_device(phaseSeriesData);
	#endif


	printf("FINISHED UNLOAD\n\n\n");
}



void dsp::CovarianceMatrix::setup_host(unsigned int chanNum, unsigned int binNum, unsigned int nPol, unsigned int nDim)
{
	_binNum = binNum;
	_freqChanNum = chanNum;
	_stokesLength = nPol * nDim;

	_covarianceMatrixLength = covariance_matrix_length(_binNum * _stokesLength);

	//Allocate memory to store the covarianceMatrix
	//upper triangle * 4 stokes vector elements
	_covarianceMatrices = new float*[_freqChanNum]; //allocate a pointer for each channel
	//_summedMeanStokesDatas = new float*[numChannels]; //allocate a pointer for each channel

	for(int i = 0; i < _freqChanNum; ++i)
	{
		//Assign the amount of memory needed for a covariance matrix in each freq channel
		_covarianceMatrices[i] = new float[ _covarianceMatrixLength ];

		//Set all the values to zero
		memset(_covarianceMatrices[i], 0, sizeof(float) * _covarianceMatrixLength);
	}


	//allocate scratch space for temporary data
	_tempMeanStokesData = new float[_binNum * _stokesLength];

	//clone the first phase series
	//_phaseSeries = new PhaseSeries(*phaseSeriesData);
}



// ------ HOST COMPUTE CODE ----------
void dsp::CovarianceMatrix::compute_covariance_matrix_host(const PhaseSeries* phaseSeriesData)
{
	//For each channel
	for(unsigned int channel = 0; channel < _freqChanNum; ++channel)
	{

		//AMPLITUDE DATA
		//IQUV, IQUV, IQUV etc etc
		const float* stokes = phaseSeriesData->get_datptr(channel, 0); //Get a pointer to the amps data


		//TODO: VINCENT, THIS COULD BE THE SOURCE OF ERRORS LATER
		const unsigned int* hits = phaseSeriesData->get_hits(0); //Get a pointer to the hit data


		//TODO: VINCENT, THIS COULD BE THE SOURCE OF ERRORS LATER RELATED TO ABOVE
		//normalise the stokes data for this freq channel
		mean_stokes_data_host(stokes, hits);


		//compute the covariance matrix
		compute_covariance_matrix_host(channel);
	}

	printf("Covariance Matrix Computed\n");

}


#if HAS_CUDA

void dsp::CovarianceMatrix::setup_device(unsigned int chanNum, unsigned int binNum, unsigned int nPol, unsigned int nDim)
{
	_binNum = binNum;
	_freqChanNum = chanNum;
	_stokesLength = nPol * nDim;

	_covarianceMatrixLength = covariance_matrix_length(_binNum * _stokesLength);

	//Allocate memory to store the covarianceMatrix
	//upper triangle * 4 stokes vector elements

	//Allocate paged locked host memory for the pointer
	cudaMallocHost(&_covarianceMatrices, sizeof(float*));


	for(int i = 0; i < _freqChanNum; ++i)
	{
		//Assign the amount of paged locked memory needed for a covariance matrix in each freq channel
		cudaMallocHost(&(_covarianceMatrices[i]), sizeof(float) * _covarianceMatrixLength);

		//Set all the values to zero
		memset(_covarianceMatrices[i], 0, sizeof(float) * _covarianceMatrixLength);
	}


	cudaMalloc(&_d_amps, sizeof(float) * _binNum * _stokesLength);
	cudaMalloc(&_d_hits, sizeof(float) * _binNum );

	//Allocate scratch space for temporary stokes data on the device
	//cudaMalloc(&_d_tempMeanStokesData, sizeof(float) * _binNum * _stokesLength);


	//Allocate space for all result vectors
	cudaMalloc(&_d_resultVector, sizeof(float) * _covarianceMatrixLength  * _freqChanNum);
}



void dsp::CovarianceMatrix::compute_covariance_matrix_device(const PhaseSeries* phaseSeriesData)
{
	/*
	computeCovarianceMatrixCUDA(float* d_resultVector, unsigned int resultByteOffset,
			float* h_amps, float* d_amps, unsigned int ampsLength,
			 float* h_hits, float* d_hits, unsigned int hitsLength,
			 unsigned int stokesLength, unsigned int blockDim2D = 16)
			 */

	for(int channel = 0; channel < _freqChanNum; ++i)
	{
		const float* h_amps = phaseSeriesData->get_datprt(channel, 0);
		const float* h_hits = phaseSeriesData->get_hits(0); //TODO: VINCENT, THIS COULD BE THE SOURCE OF ERRORS LATER

		computeCovarianceMatrixCuda
			(_d_resultVector, i * _covarianceMatrixLength * sizeof(float),
			h_amps, _d_amps, _binNum * _stokesLength,
			h_hits, _d_hits, _binNum, _stokesLength);
	}

}


#endif



void dsp::CovarianceMatrix::compute_covariance_matrix_host(unsigned int freqChan)
{
	for(int row = 0; row < _binNum; ++row)
	{
		for(int col = row; col < _binNum; ++col)
		{
			_covarianceMatrices[freqChan][ (row * _binNum + col) - covariance_matrix_length(row) ] +=
					_tempMeanStokesData[row] * _tempMeanStokesData[col];
		}
	}


}



void dsp::CovarianceMatrix::mean_stokes_data_host(const float* stokesData, const unsigned int* hits)
{
	int totalLength = _binNum * _stokesLength;

	for(int i = 0; i < totalLength; ++i)
	{
		_tempMeanStokesData[ i ] = stokesData[ i ] / hits[ i / 4 ];
	}
}



unsigned int dsp::CovarianceMatrix::covariance_matrix_length(const unsigned int numBin)
{
	return (numBin * (numBin + 1)) / 2;
}



void dsp::CovarianceMatrix::set_unloader(PhaseSeriesUnloader* unloader)
{
	_unloader = unloader;
}

