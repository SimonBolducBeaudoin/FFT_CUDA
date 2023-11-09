#include"FFT_CUDA.h"

void FFT_CUDA(int n, std::complex<float>* in)
{
	cufftHandle plan;
	if (cufftPlan1d(&plan, n, CUFFT_C2C, 1) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation failed");
	}
	if (cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(in), 
							reinterpret_cast<cufftComplex*>(in), 
							CUFFT_FORWARD) != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		throw std::runtime_error("CUFFT error: ExecC2C Forward failed");
	}
	cufftDestroy(plan);
}

void rFFT_CUDA(int n, std::complex<float>* in)
{
	cufftHandle plan;
	if (cufftPlan1d(&plan, n, CUFFT_R2C, 1) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation failed");
	}
	if (cufftExecR2C(plan, reinterpret_cast<cufftReal*>(in), 
							reinterpret_cast<cufftComplex*>(in)) != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		throw std::runtime_error("CUFFT error: ExecR2C failed");
	}
	cufftDestroy(plan);
}

void rFFT_Block_CUDA(int n, int size, float* in,std::complex<float>* out)
{

	int rank = 1; // Dimensionality
	long long int batch = n/size; // How many dft to compute
	long long int length[] = {size}; // Length of each dimension
	long long int inembed[] = {size*batch}; // Length of input dimensions 
	long long int onembed[] = {(size/2+1)*batch}; // Length of output dimensions 
	long long int idist = size; // Distance between dfts in
	long long int odist = size/2+1; // Distance between dfts out
	long long int stride = 1; // Stride
	size_t worksize[1]; // Number of gpus

	cufftHandle plan;
	if(cufftCreate(&plan) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation failed");
	}

	if (cufftMakePlanMany64(plan, rank, length, inembed,
        stride, idist, onembed, stride,
        odist, CUFFT_R2C, batch, worksize) != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		throw std::runtime_error("CUFFT error: Plan initialization failed");
	}
	if (cufftExecR2C(plan, reinterpret_cast<cufftReal*>(in), reinterpret_cast<cufftComplex*>(out)) != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		throw std::runtime_error("CUFFT error: ExecR2C failed");
	}
	cufftDestroy(plan);
}