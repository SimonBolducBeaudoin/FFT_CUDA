#include "FFT_CUDA_py.h"

void init_fft(py::module &m)
{
	m.def("fft_cuda",&FFT_CUDA_py<float>,"in"_a);
}

PYBIND11_MODULE(libfftcuda, m)
{
	m.doc() = "Might work, might not.";
	init_fft(m);
}
