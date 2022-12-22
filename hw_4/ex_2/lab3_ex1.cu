
#include <stdio.h>
#include <sys/time.h>
#include <random>
#include <vector>

#define DataType double

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  out[i] = in1[i] + in2[i];
}


//@@ Insert code to implement timer start

clock_t CLOCK;

void start_clock() {
    CLOCK = clock();
}

//@@ Insert code to implement timer stop

double stop_clock() {
    clock_t stop_time = clock();
    return ((double) (stop_time - CLOCK)) / CLOCKS_PER_SEC;
}

__host__ int asMultipleOf(int value, int factor) {
    // Returns the smallest value larger or equal to 
    // "value" that is a multiple of "factor"
    int rest = value % factor;
    return rest == 0 ? value : (value + factor - rest);
}


int main(int argc, char **argv) {
  
  int inputLength;
  int deviceLength;
  double elapsed; 
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

  deviceLength = asMultipleOf(inputLength, TPB);

  int blocks = deviceLength / TPB;

  printf("The input length is   : %d\n", inputLength);
  printf("The device length is  : %d\n", deviceLength);
  
  // Allocate Host memory for input and output
  hostInput1 = (DataType*) malloc(inputLength * sizeof *hostInput1);
  hostInput2 = (DataType*) malloc(inputLength * sizeof *hostInput2);
  hostOutput = (DataType*) malloc(inputLength * sizeof *hostOutput);
  resultRef = (DataType*) malloc(inputLength * sizeof *resultRef);
  
  
  // Initialize hostInput1 and hostInput2 to random numbers.

  std::normal_distribution<DataType> distribution(0.0, 10.0);
  std::default_random_engine gen(1337);
  
  for (DataType *ptr : {hostInput1, hostInput2}) {
      for (int i=0; i<inputLength; ++i){
          ptr[i] = distribution(gen);
      }
  }
  
  start_clock();
  // Create reference result in CPU
  for (int i=0; i < inputLength; ++i){
      resultRef[i] = hostInput1[i] + hostInput2[i];
  }
  elapsed = stop_clock();
  printf("host execution time   : %f\n", elapsed);


  // Allocate GPU memory
  
  start_clock();
  cudaMalloc(&deviceInput1, deviceLength * sizeof*deviceInput1);
  cudaMalloc(&deviceInput2, deviceLength * sizeof*deviceInput2);
  cudaMalloc(&deviceOutput, deviceLength * sizeof*deviceOutput);
  elapsed = stop_clock();
  printf("cudamalloc            : %f\n", elapsed);


  // Copy memory to the GPU
  
  start_clock();
  cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof *hostInput1, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof *hostInput2, cudaMemcpyHostToDevice);
  elapsed = stop_clock();
  printf("host to device memcpy : %f\n", elapsed);



  // Launch the GPU Kernel
  start_clock();
  vecAdd<<<blocks, TPB>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
  cudaDeviceSynchronize();
  elapsed = stop_clock();
  printf("kernel execution time : %f\n", elapsed);

  // Copy the GPU memory back to the CPU

  start_clock();
  cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof*hostOutput, cudaMemcpyDeviceToHost);
  elapsed = stop_clock();
  printf("device to host memcpy : %f\n", elapsed); 

  // Compare the output with the reference

  DataType m = hostOutput[0] - resultRef[0];
  DataType m2 = 0.0;

  DataType maximumError = 0.0;

  for (int i=1.0; i<inputLength; ++i) {
    DataType delta;
    DataType weight;
    DataType diff;

    weight = DataType(i);
    diff = hostOutput[i] - resultRef[i];

    delta = diff - m;

    m += delta / (weight + 1.0);
    m2 += pow(delta, 2) * weight / (weight + 1.0);
    maximumError = max(abs(diff), maximumError);
  }

  m2 = sqrt(m2 / DataType(inputLength-1));

  printf("mean difference       : %f\n", m);
  printf("standard deviation    : %f\n", m2);
  printf("maximum error         : %f\n", maximumError);
  
  // Free the GPU memory

  for (DataType *ptr : {deviceInput1, deviceInput2, deviceOutput}) {
    cudaFree(ptr);
  }

  // Free the CPU memory here

  for (DataType *ptr : {hostInput1, hostInput2, hostOutput, resultRef}) {
    free(ptr);
  }
  return 0;
}
