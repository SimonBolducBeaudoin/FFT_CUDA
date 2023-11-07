template<class DataType>
py::array_t<std::complex<DataType>, py::array::c_style> FFT_CUDA_py(
				py::array_t<std::complex<DataType>, py::array::c_style> py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;
	
	std::complex<DataType>* ptr_py_in = (std::complex<DataType>*) buf_in.ptr;
	
	std::complex<DataType>* out;
	out = (std::complex<DataType>*) malloc(n*sizeof(std::complex<DataType>));
	
	std::complex<DataType>* gpu;
	cudaMalloc((void**)&gpu, 2*n*sizeof(DataType));
	
	cudaMemcpy(gpu,ptr_py_in,2*sizeof(DataType)*n,cudaMemcpyHostToDevice);
	
	FFT_CUDA(n, gpu);

	cudaMemcpy(out,gpu,2*sizeof(DataType)*n,cudaMemcpyDeviceToHost);
	cudaFree(gpu);

	py::capsule free_when_done( out, free );
	return py::array_t<std::complex<DataType>, py::array::c_style> 
	(
		{n},
		{2*sizeof(DataType)},
		out,
		free_when_done	
	);
}