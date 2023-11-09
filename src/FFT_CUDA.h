#pragma once
#include <cufft.h>
#include <complex>
// #include <string>
// #include <stdio.h>

void FFT_CUDA(int n, std::complex<float>* in);
void rFFT_CUDA(int n, std::complex<float>* in);
void rFFT_Block_CUDA(int n, int size, float* in,std::complex<float>* out);