
#include <stdio.h>
#include <sys/time.h>
#include <random>

#define DataType double

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  out[i] = in1[i] + in2[i];
}

//@@ Insert code to implement timer start

//@@ Insert code to implement timer stop



__host__ int asMultipleOf(int value, int factor) {
    // Returns the smallest value larger or equal to 
    // "value" that is a multiple of "factor"
    int rest = value % factor;
    return rest == 0 ? value : (value + factor - rest);
}


int main(int argc, char **argv) {
  
  int inputLength;
  int deviceLength;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;
  

  // Initialize the 1D grid and block dimensions here

  // Read in inputLength from args
  inputLength = atoi(argv[1]);


  int TPB = 32;

  deviceLength = 132; //asMultipleOf(inputLength, TPB);

  int blocks = deviceLength / TPB;

  printf("The input length is %d\n", inputLength);
  printf("The device length is %d\n", deviceLength);
  
  // Allocate Host memory for input and output
  hostInput1 = (DataType*) malloc(inputLength * sizeof *hostInput1);
  hostInput2 = (DataType*) malloc(inputLength * sizeof *hostInput2);
  hostOutput = (DataType*) malloc(inputLength * sizeof *hostOutput);
  resultRef = (DataType*) malloc(inputLength * sizeof *resultRef);
  
  
  // Initialize hostInput1 and hostInput2 to random numbers.

  std::normal_distribution<DataType> distribution(0.0, 1.0);
  std::default_random_engine gen(1337);
  
  for (DataType *ptr : {hostInput1, hostInput2}) {
      for (int i=0; i<inputLength; ++i){
          ptr[i] = distribution(gen);
      }
  }

  // Create reference result in CPU
  for (int i=0; i < inputLength; ++i){
      resultRef[i] = hostInput1[i] + hostInput2[i];
  }

  // Allocate GPU memory

  cudaMalloc(&deviceInput1, deviceLength * sizeof*deviceInput1);
  cudaMalloc(&deviceInput2, deviceLength * sizeof*deviceInput2);
  cudaMalloc(&deviceOutput, deviceLength * sizeof*deviceOutput);


  // Copy memory to the GPU

  cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof *hostInput1, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof *hostInput2, cudaMemcpyHostToDevice);



  //@@ Launch the GPU Kernel here
    
  vecAdd<<<blocks, TPB>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);

  //@@ Copy the GPU memory back to the CPU here

  cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof*hostOutput, cudaMemcpyDeviceToHost);


  //@@ Insert code below to compare the output with the reference

  double delta = 0.0;

  for (int i=0; i<inputLength; ++i){
    delta += pow(hostOutput[i] - resultRef[i], 2);
  }

  printf("euclidean distance between ref and out: %f\n", delta);

  //@@ Free the GPU memory here

  for (DataType *ptr : {deviceInput1, deviceInput2, deviceOutput}) {
    cudaFree(ptr);
  }

  //@@ Free the CPU memory here

  for (DataType *ptr : {hostInput1, hostInput2, hostOutput}) {
    free(ptr);
  }
  return 0;
}
