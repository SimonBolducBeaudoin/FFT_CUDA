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

