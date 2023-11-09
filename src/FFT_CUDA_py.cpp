#include "FFT_CUDA_py.h"

void init_fft(py::module &m)
{
	m.def("fft_cuda",&FFT_CUDA_py<float>,"in"_a.noconvert());
    m.def("rfft_cuda",&rFFT_CUDA_py<float>, "in"_a.noconvert());
    m.def("rfft_cuda",&rFFT_Block_CUDA_py<float>, "in"_a.noconvert(),"in"_a);
    
    
}

PYBIND11_MODULE(fftcuda, m)
{
	m.doc() = "Might work, might not.";
	init_fft(m);
}
