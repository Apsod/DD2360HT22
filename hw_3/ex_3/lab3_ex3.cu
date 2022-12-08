#include <stdio.h>
#include <sys/time.h>
#include <random>

#define NUM_BINS 4096

__host__  __device__ int divUp(int numerator, int denominator) {
    // Returns numerator / denominator rounded up. 
    return (numerator + denominator - 1) / denominator;
}

__host__  __device__ int asMultipleOf(int value, int factor) {
    // Returns the smallest value larger or equal to 
    // "value" that is a multiple of "factor"
    // int rest = value % factor;
    return factor * divUp(value, factor); 
}

__global__ void kernel2(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {
  extern __shared__ unsigned int val[];
  const int tid =  threadIdx.x; 
  const int num = blockIdx.y;
  const int group_size = blockDim.x; 
  const int input_start = blockIdx.x * group_size + tid;
  const int input_end = min(input_start + group_size, num_elements);

  val[tid] = 0; 
  
  for (int i=input_start; i<input_end; i+=group_size){
      if (input[i] == num){
          val[tid] += 1;
      }
  }

  for(int i = 1; i < group_size; i = i << 1){
    __syncthreads();
    if (tid % (i << 1) == 0 and tid+i < group_size) {
      val[tid] += val[tid+i];
    }
  }
  if (tid == 0 and val[0] != 0){
      atomicAdd(bins + num, val[0]); 
  }
}

__global__ void kernel1(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {
  extern __shared__ unsigned int val[];
  const int tid = threadIdx.x;
  const int group_size = blockDim.x; 
  const int input_i = blockIdx.x * group_size + tid; 
  
  const int bin_group = NUM_BINS / group_size;
  const int bin_start = bin_group * tid;
  const int bin_end = min(bin_start + bin_group, num_bins);

  for (int i=bin_start; i<bin_end; ++i){
      val[i] = 0; 
  }

  __syncthreads();
  
  if (input_i < num_elements) {
    atomicAdd(val + input[input_i], 1);
  }

  __syncthreads();

  for (int i=bin_start; i<bin_end; ++i){
    if (val[i] > 0) {
      atomicAdd(bins + i, val[i]); 
    }
  }
}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < num_bins){
      bins[index] = min(bins[index], 127);
  }
}


int main(int argc, char **argv) {
  
  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *resultRef;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

  //@@ Insert code below to read in inputLength from args

  inputLength = atoi(argv[1]);

  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output


  hostInput = (unsigned int*) malloc(inputLength * sizeof *hostInput);
  hostBins = (unsigned int*) malloc(NUM_BINS * sizeof *hostBins);
  resultRef = (unsigned int*) malloc(NUM_BINS * sizeof *resultRef);
  memset(hostBins, 0, NUM_BINS * sizeof *hostBins); 
  memset(resultRef, 0, NUM_BINS * sizeof *resultRef); 

  
  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)

  std::default_random_engine gen(1337);
  std::uniform_int_distribution<unsigned int> distribution(0, NUM_BINS-1);

  for (int i=0; i<inputLength; ++i){
      hostInput[i] = distribution(gen);
  }
  
  //@@ Insert code below to create reference result in CPU

  for (int i=0; i<inputLength; ++i){
      int j = hostInput[i];
      if (resultRef[j] < 127) {
          resultRef[j] += 1; 
      }
  }


  //@@ Insert code below to allocate GPU memory here

  cudaMalloc(&deviceInput, inputLength * sizeof *deviceInput);
  cudaMalloc(&deviceBins, NUM_BINS * sizeof *deviceBins);

  //@@ Insert code to Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, inputLength * sizeof *deviceInput, cudaMemcpyHostToDevice);

  //@@ Insert code to initialize GPU results

  cudaMemset(deviceBins, 0, NUM_BINS * sizeof *hostBins); 


  //@@ Initialize the grid and block dimensions here

  //int TPB = asMultipleOf(min(inputLength, 1024), 32);
  //dim3 BLOCKS(divUp(inputLength, TPB), NUM_BINS);

  //printf("binning: B: (%d, %d) TPB: %d\n", BLOCKS.x, BLOCKS.y, TPB);

  //kernel2<<<BLOCKS, TPB, TPB * sizeof *deviceBins>>>(deviceInput, deviceBins, inputLength, NUM_BINS);
  //cudaDeviceSynchronize();


  int TPB = asMultipleOf(min(inputLength, 1024), 32);
  int BLOCKS = divUp(inputLength, TPB);
  
  printf("binning: B: %d TPB: %d\n", BLOCKS, TPB);

  kernel1<<<BLOCKS, TPB, NUM_BINS * sizeof *deviceBins>>>(deviceInput, deviceBins, inputLength, NUM_BINS);
  cudaDeviceSynchronize();

  TPB = 32; //asMultipleOf(min(NUM_BINS, 1024), 32);
  int BLOCKS2 = divUp(NUM_BINS, TPB); 

  printf("converting: B: %d TPB: %d\n", BLOCKS2, TPB);

  convert_kernel<<<BLOCKS2, TPB>>>(deviceBins, NUM_BINS); 
  cudaDeviceSynchronize();

  //@@ Copy the GPU memory back to the CPU here

  cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof *hostBins, cudaMemcpyDeviceToHost);

  //@@ Insert code below to compare the output with the reference

  for (int i=0; i<NUM_BINS; ++i){
      if (resultRef[i] !=  hostBins[i]) {
          printf("mismatch in %d, %d != %d\n", i, resultRef[i], hostBins[i]); 
      }
  }

  for (unsigned int *ptr : {deviceInput, deviceBins}) {
    cudaFree(ptr);
  }

  // Free the CPU memory

  for (unsigned int *ptr : {hostInput, hostBins, resultRef}) {
    free(ptr);
  }
  return 0;
}

