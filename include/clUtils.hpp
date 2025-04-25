#pragma once
#ifndef CL_UTILS_HPP
#define CL_UTILS_HPP

#include <vector>
#include <string>
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>

// helper function to check OpenCL errors
const char *getCLErrorString(cl_int error);

// helper function to read OpenCL kernel source file
const std::string readKernelSource(const std::string& source_path);

class clUtils {
public:
  clUtils();
  ~clUtils();

  void showCLInfo();

  void setPlatformAndDevice();
  void setPlatformAndDevice(int platform_id, int device_id);

  void buildKernels(const char *kernel_dir_path);
  
  cl::Context cl_context;
  cl::CommandQueue cl_cmdqueue;
  cl::Program cl_program;
private:
  std::vector<cl::Platform> cl_platforms;
  std::vector<std::vector<cl::Device>> cl_devices;
  cl::Platform selected_platform;
  cl::Device selected_device;

  
};


#endif
