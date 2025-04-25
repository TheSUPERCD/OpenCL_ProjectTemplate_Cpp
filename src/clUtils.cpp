#include <iostream>
#include <fstream>
#include <filesystem>
#include <stdexcept>

#include <clUtils.hpp>

clUtils::clUtils(){
  cl_int err = cl::Platform::get(&cl_platforms);
  if(cl_platforms.empty()){
    std::cerr << "\nERROR: No OpenCL platforms detected!\n";
    throw std::runtime_error(getCLErrorString(err));
  } else {
    int gpu_count = 0;
    for(int i=0;i<cl_platforms.size();i++){
      std::vector<cl::Device> devices;
      cl_platforms[i].getDevices(CL_DEVICE_TYPE_GPU, &devices);
      cl_devices.push_back(devices);
      gpu_count += devices.size();
    }

    if(gpu_count == 0){
      throw std::runtime_error("\nERROR: No OpenCL-compatible GPU devices found!");
    }
  }
}

clUtils::~clUtils(){

}

void clUtils::showCLInfo(){
  for(int i=0;i<cl_platforms.size();i++){
    std::cout << "Detected OpenCL Platform " << i << " : " << cl_platforms[i].getInfo<CL_PLATFORM_NAME>() << std::endl;
    std::cout << "  - Platform vendor   : " << cl_platforms[i].getInfo<CL_PLATFORM_VENDOR>() << std::endl;
    std::cout << "  - Platform version  : " << cl_platforms[i].getInfo<CL_PLATFORM_VERSION>() << std::endl;
    std::cout << "  - Number of Devices : " << cl_devices[i].size() << std::endl;
    for(int j=0;j<cl_devices[i].size();j++){
      std::cout << "      - Device["<< j <<"] Name : " << cl_devices[i][j].getInfo<CL_DEVICE_NAME>() << std::endl;
      std::cout << "         - Max Compute Units   : " << cl_devices[i][j].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
      std::cout << "         - Max Clock Frequency : " << cl_devices[i][j].getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() <<"MHz"<< std::endl;
      std::cout << "         - Max Work Group Size : " << cl_devices[i][j].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
      std::cout << "         - Max Work-item Dim   : " << cl_devices[i][j].getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>() << std::endl;
    }
  }
}

void clUtils::setPlatformAndDevice(){
  int valid_platform_id = 0;
  while(!cl_devices[valid_platform_id].size()){
    valid_platform_id++;
  }
  selected_platform = cl_platforms[valid_platform_id];
  selected_device = cl_devices[valid_platform_id][0];
}

void clUtils::setPlatformAndDevice(int platform_id, int device_id){
  if(platform_id>=0 && platform_id<cl_platforms.size()){
    selected_platform = cl_platforms[platform_id];
    if(device_id>=0 && device_id<cl_devices[platform_id].size()){
      selected_device = cl_devices[platform_id][device_id];
    } else {
      throw std::runtime_error("\nERROR: clUtils::device_id out of range!");
    }
  } else {
    throw std::runtime_error("\nERROR: clUtils::platform_id out of range!");
  }
}


void clUtils::buildKernels(const char *kernel_dir_path){
  cl_context = cl::Context(selected_device);
  cl_cmdqueue = cl::CommandQueue(cl_context, selected_device);
  
  cl::Program::Sources cl_kernel_sources;
  for(const auto& entry : std::filesystem::directory_iterator(kernel_dir_path)){
    std::string kernel_source_filepath = entry.path().string();
    std::string kernel_source_code = readKernelSource(kernel_source_filepath);
    
    cl_kernel_sources.push_back({kernel_source_code.c_str(), kernel_source_code.length()});
  }

  cl_program = cl::Program(cl_context, cl_kernel_sources);
  try {
    cl_program.build({selected_device});
  } catch(const cl::BuildError& e) {
    std::cerr << "\nERROR: CL_PROGRAM_BUILD_ERROR | Status: " << e.err() << "\n" 
      << "Build Log: \n" << cl_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(selected_device) << "\n";
    throw std::runtime_error("\nERROR: CL_PROGRAM_BUILD_ERROR occurred!");
  }
}









const std::string readKernelSource(const std::string& source_path){
  std::ifstream clKernelFile(source_path, std::ios::binary | std::ios::in | std::ios::ate);
  if(!clKernelFile.is_open()){
    throw std::runtime_error("\nERROR: Unable to open kenel source file!");
  }
  
  // enable exceptions
  clKernelFile.exceptions(std::ios::failbit | std::ios::badbit);
  
  std::string file_content;
  size_t file_size = clKernelFile.tellg();
  file_content.resize(file_size);
  clKernelFile.seekg(0, std::ios::beg);
  if(!clKernelFile.read(&file_content[0], file_size)){
    throw std::runtime_error("\nERROR: Unable to read kernel source file!");
  }

  return file_content;
}

const char *getCLErrorString(cl_int error) {
    switch (error) {
        case CL_SUCCESS:                            return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND:                   return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE:               return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE:             return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES:                   return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:                 return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE:       return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP:                   return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH:              return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE:              return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE:                        return "CL_MAP_FAILURE";
        case CL_INVALID_VALUE:                      return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE:                return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM:                   return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE:                     return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT:                    return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES:           return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE:              return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR:                   return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT:                 return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE:                 return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER:                    return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY:                     return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS:              return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM:                    return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE:         return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME:                return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION:          return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL:                     return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX:                  return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE:                  return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE:                   return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS:                return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION:             return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE:            return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE:             return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET:              return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST:            return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT:                      return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION:                  return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT:                  return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE:                return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL:                  return "CL_INVALID_MIP_LEVEL";
        case CL_INVALID_GLOBAL_WORK_SIZE:           return "CL_INVALID_GLOBAL_WORK_SIZE";
        default:                                    return "Unknown OpenCL error";
    }
}

