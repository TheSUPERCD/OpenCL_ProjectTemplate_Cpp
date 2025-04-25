#include <iostream>
#include <clUtils.hpp>



int main (int argc, char *argv[]) {
  clUtils clEngine;
  clEngine.showCLInfo();

  const int arr_len = 100;
  std::vector<int> A;
  std::vector<int> B;
  std::vector<int> C;
  
  for(int i=0;i<100;i++){
    A.push_back(i+1);
    B.push_back((i+1)*2);
  }

  try {
    clEngine.setPlatformAndDevice();
    clEngine.buildKernels("../kernels/");
    
    cl::Buffer d_A(clEngine.cl_context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, arr_len*sizeof(int), A.data());
    cl::Buffer d_B(clEngine.cl_context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, arr_len*sizeof(int), B.data());
    cl::Buffer d_C(clEngine.cl_context, CL_MEM_WRITE_ONLY, arr_len*sizeof(int));

    cl::Kernel cl_kernel(clEngine.cl_program, "vector_add");
    cl_kernel.setArg(0, d_A);
    cl_kernel.setArg(1, d_B);
    cl_kernel.setArg(2, d_C);
    cl_kernel.setArg(3, 100);

    cl::Event cl_event;
    cl::NDRange global_size(100,1);
    cl::NDRange local_size = cl::NullRange;
    clEngine.cl_cmdqueue.enqueueNDRangeKernel(cl_kernel, cl::NullRange, global_size, local_size, nullptr, &cl_event);
    cl_event.wait();
    
    C.resize(arr_len);
    clEngine.cl_cmdqueue.enqueueReadBuffer(d_C, CL_TRUE, 0, 100*sizeof(int), &C[0]);
  } catch (std::runtime_error& e) {
    std::cerr << e.what() << std::endl;
  }

  for(int i=0;i<100;i++){
    std::cout << C[i] << ", ";
  }
  std::cout << std::endl;
  return 0;
}
