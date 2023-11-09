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

template< class DataType >
py::array_t<std::complex<DataType>, py::array::c_style> rFFT_CUDA_py(py::array_t<DataType, py::array::c_style> py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;
	
	DataType* ptr_py_in = (DataType*) buf_in.ptr;
	
	std::complex<DataType>* out;
	out = (std::complex<DataType>*) malloc((n+2)*sizeof(DataType));

	std::complex<DataType>* gpu;
	cudaMalloc((void**)&gpu, (n/2+1)*sizeof(std::complex<DataType>));
	
	cudaMemcpy(gpu,ptr_py_in,sizeof(DataType)*n,cudaMemcpyHostToDevice);
	
	rFFT_CUDA(n,gpu);

	cudaMemcpy(out,gpu,(n+2)*sizeof(DataType),cudaMemcpyDeviceToHost);

	cudaFree(gpu);

	py::capsule free_when_done( out, free );
	return py::array_t<std::complex<DataType>, py::array::c_style> 
	(
		{n/2+1},
		{sizeof(std::complex<DataType>)},
		out,
		free_when_done	
	);
}

template< class DataType >
py::array_t<std::complex<DataType>, py::array::c_style> rFFT_Block_CUDA_py(py::array_t<DataType, py::array::c_style> py_in, int l_fft)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;
    int batch = n/l_fft;
	
	DataType* ptr_py_in = (DataType*) buf_in.ptr;
	
	std::complex<DataType>* out;
	out = (std::complex<DataType>*) malloc(batch*(l_fft/2+1)*sizeof(std::complex<DataType>));

	DataType* gpu_in;
	cudaMalloc((void**)&gpu_in, n*sizeof(DataType));
    
    cudaMemcpy(gpu_in,ptr_py_in,sizeof(DataType)*n,cudaMemcpyHostToDevice);
    
    std::complex<DataType>* gpu_out;
	cudaMalloc((void**)&gpu_out, batch*(l_fft/2+1)*sizeof(std::complex<DataType>));
	
    rFFT_Block_CUDA(n,l_fft,gpu_in,gpu_out);

	cudaMemcpy(out,gpu_out,batch*(l_fft/2+1)*sizeof(std::complex<DataType>),cudaMemcpyDeviceToHost);

	cudaFree(gpu_in);
    cudaFree(gpu_out);

	py::capsule free_when_done( out, free );
	return py::array_t<std::complex<DataType>, py::array::c_style> 
	(
		{batch,l_fft},
		{(l_fft/2+1)*sizeof(std::complex<DataType>),sizeof(std::complex<DataType>)},
		out,
		free_when_done	
	);
}
