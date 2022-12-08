
#include <stdio.h>
#include <sys/time.h>
#include <random>

#define NUM_BINS 4096

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {

//@@ Insert code below to compute histogram of input using shared memory and atomics


}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {

//@@ Insert code below to clean up bins that saturate at 127


}


int main(int argc, char **argv) {
  
  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *resultRef;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

  //@@ Insert code below to read in inputLength from args

  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output

  
  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)


  //@@ Insert code below to create reference result in CPU


  //@@ Insert code below to allocate GPU memory here


  //@@ Insert code to Copy memory to the GPU here


  //@@ Insert code to initialize GPU results


  //@@ Initialize the grid and block dimensions here


  //@@ Launch the GPU Kernel here


  //@@ Initialize the second grid and block dimensions here


  //@@ Launch the second GPU Kernel here


  //@@ Copy the GPU memory back to the CPU here


  //@@ Insert code below to compare the output with the reference


  //@@ Free the GPU memory here


  //@@ Free the CPU memory here


  return 0;
}

