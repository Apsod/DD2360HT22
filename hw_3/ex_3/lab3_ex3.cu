#include <stdarg.h>
#include <stdio.h>
#include <sys/time.h>
#include <random>

#define NUM_BINS 4096

#ifndef STRIDE
#define STRIDE 8
#endif

#ifdef VERBOSE
//COPIED FROM STACK OVERFLOW
void myprint(const char *fmt, ...){
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
}
#endif
#ifndef VERBOSE
void myprint(const char *format, ...){}
#endif

clock_t CLOCK;

void start_clock() {
    CLOCK = clock();
}

//@@ Insert code to implement timer stop

double stop_clock() {
    clock_t stop_time = clock();
    return ((double) (stop_time - CLOCK)) / CLOCKS_PER_SEC;
}

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
  const int block = blockIdx.x; 
  const int tpb = blockDim.x; 

  const int block_chunk = tpb * STRIDE;

  const int input_start = block * block_chunk + tid;
  const int input_end = min(input_start + block_chunk, num_elements); 
  
  val[tid] = 0;

  
  for (int i=input_start; i<input_end; i += tpb){
    if (input[i] == num) {
        val[tid] +=1;
    }
  }

  for(int i = 1; i < tpb; i = i << 1){
    __syncthreads();
    if (tid % (i << 1) == 0 and tid+i < tpb) {
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
  const int tpb = blockDim.x; 
  const int block = blockIdx.x;

  const int block_chunk = tpb * STRIDE;

  const int input_start = block * block_chunk + tid; 
  const int input_end = min(input_start + block_chunk, num_elements); 
  
  const int bin_chunk = NUM_BINS / tpb;
  const int bin_start = bin_chunk * tid;
  const int bin_end = min(bin_start + bin_chunk, num_bins);

  for (int i=bin_start; i<bin_end; ++i){
      val[i] = 0; 
  }

  __syncthreads();
  
  for (int i=input_start; i < input_end; i+=tpb){
    atomicAdd(val + input[i], 1);
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

  myprint("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output


  hostInput = (unsigned int*) malloc(inputLength * sizeof *hostInput);
  hostBins = (unsigned int*) malloc(NUM_BINS * sizeof *hostBins);
  resultRef = (unsigned int*) malloc(NUM_BINS * sizeof *resultRef);
  memset(hostBins, 0, NUM_BINS * sizeof *hostBins); 
  memset(resultRef, 0, NUM_BINS * sizeof *resultRef); 

  
  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
  
#ifndef CONGESTION
  std::default_random_engine gen(1337);
  std::uniform_int_distribution<unsigned int> distribution(0, NUM_BINS-1);

  for (int i=0; i<inputLength; ++i){
      hostInput[i] = distribution(gen);
  }
#endif
#ifdef CONGESTION
  for (int i=0; i<inputLength; ++i){
      hostInput[i] = 0; 
  }
#endif

  
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
  myprint("stride: %d\n", STRIDE);
#ifdef ALTERNATIVE_KERNEL
  int TPB = asMultipleOf(min(inputLength, 1024), 32);
  dim3 BLOCKS(divUp(inputLength, TPB*STRIDE), NUM_BINS);

  myprint("binning: B: (%d, %d) TPB: %d\n", BLOCKS.x, BLOCKS.y, TPB);

  kernel2<<<BLOCKS, TPB, TPB * sizeof *deviceBins>>>(deviceInput, deviceBins, inputLength, NUM_BINS);
  cudaDeviceSynchronize();
#endif  
#ifndef ALTERNATIVE_KERNEL
  int TPB = asMultipleOf(min(inputLength, 1024), 32);
  int BLOCKS = divUp(inputLength, TPB*STRIDE);
  
  myprint("binning: B: %d TPB: %d\n", BLOCKS, TPB);

  kernel1<<<BLOCKS, TPB, NUM_BINS * sizeof *deviceBins>>>(deviceInput, deviceBins, inputLength, NUM_BINS);
  cudaDeviceSynchronize();
#endif



  cudaDeviceSynchronize();

  TPB = 32; //asMultipleOf(min(NUM_BINS, 1024), 32);
  int BLOCKS2 = divUp(NUM_BINS, TPB); 

  myprint("converting: B: %d TPB: %d\n", BLOCKS2, TPB);

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

#ifdef OUTPUT
  for (int i=0; i < NUM_BINS; ++i){
      printf("%d ", hostBins[i]);
  }
  printf("\n");
#endif

  for (unsigned int *ptr : {deviceInput, deviceBins}) {
    cudaFree(ptr);
  }

  // Free the CPU memory

  for (unsigned int *ptr : {hostInput, hostBins, resultRef}) {
    free(ptr);
  }
  return 0;
}

