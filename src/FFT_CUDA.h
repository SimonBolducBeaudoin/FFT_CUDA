#pragma once
#include <cufft.h>
#include <complex>
// #include <string>
// #include <stdio.h>

void FFT_CUDA(int n, std::complex<float>* in);
