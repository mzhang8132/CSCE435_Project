/*
 * Parallel bitonic sort using CUDA.
 * Compile with
 * nvcc bitonic_sort.cu
 * Based on http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm
 * License: BSD 3
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

int THREADS;
int BLOCKS;
int NUM_VALS;
int OPTION;

const char* options[3] = {"random", "sorted", "reverse_sorted"};

float random_float()
{
  return (float)rand()/(float)RAND_MAX;
}

void array_print(float *arr, int length) 
{
  int i;
  for (i = 0; i < length; ++i) {
    printf("%1.3f ",  arr[i]);
  }
  printf("\n");
}

void array_fill(float *arr, int length, int option) {
    if (option == 1) {
        srand(0);
        for (int i = 0; i < length; ++i) {
            arr[i] = random_float();
        }
    } else if (option == 2) {
        for (int i = 0; i < length; ++i) {
            arr[i] = (float)i;
        }
    } else if (option == 3) {
        for (int i = 0; i < length; ++i) {
            arr[i] = (float)length-1-i;
        }
    }
}

__global__ void bitonic_sort_step(float *dev_values, int j, int k)
{
  unsigned int i, ixj; /* Sorting partners: i and ixj */
  i = threadIdx.x + blockDim.x * blockIdx.x;
  ixj = i^j;

  /* The threads with the lowest ids sort the array. */
  if ((ixj)>i) {
    if ((i&k)==0) {
      /* Sort ascending */
      if (dev_values[i]>dev_values[ixj]) {
        /* exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
    if ((i&k)!=0) {
      /* Sort descending */
      if (dev_values[i]<dev_values[ixj]) {
        /* exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
  }
}

/**
 * Inplace bitonic sort using CUDA.
 */
void bitonic_sort(float *values)
{
  CALI_CXX_MARK_FUNCTION;
  float *dev_values;
  size_t size = NUM_VALS * sizeof(float);

  cudaMalloc((void**) &dev_values, size);
  
  //CALI START
  CALI_MARK_BEGIN("comm");
  CALI_MARK_BEGIN("comm_large");

  //MEM COPY FROM HOST TO DEVICE
  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);
  //MEM COPY FROM HOST TO DEVICE

  //CALI END
  CALI_MARK_END("comm");
  CALI_MARK_END("comm_large");

  dim3 blocks(BLOCKS,1);    /* Number of blocks   */
  dim3 threads(THREADS,1);  /* Number of threads  */
  
  int j, k;
  //CALI START
  CALI_MARK_BEGIN("comp");
  CALI_MARK_BEGIN("comp_large");
  
  /* Major step */
  for (k = 2; k <= NUM_VALS; k <<= 1) {
    /* Minor step */
    for (j=k>>1; j>0; j=j>>1) {
      bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
      iterations++;
    }
  }
  cudaDeviceSynchronize();

  //CALI END
  CALI_MARK_END("comp");
  CALI_MARK_END("comp_large");

  //CALI START
  CALI_MARK_BEGIN("comm");
  CALI_MARK_BEGIN("comm_large");

  //MEM COPY FROM DEVICE TO HOST
  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
  //MEM COPY FROM DEVICE TO HOST

  //CALI END
  CALI_MARK_END("comm");
  CALI_MARK_END("comm_large");

  //CUDA FREE
  cudaFree(dev_values);
}

int isSorted(float *numbers, int length) {
  for(int i=1; i<length; i++) {
    if (numbers[i] < numbers[i-1]) return 0;
  }
  return 1;
}

int main(int argc, char *argv[])
{
  CALI_CXX_MARK_FUNCTION;

  THREADS = atoi(argv[1]);
  NUM_VALS = atoi(argv[2]);
  BLOCKS = NUM_VALS / THREADS;

  // printf("Number of threads: %d\n", THREADS);
  // printf("Number of values: %d\n", NUM_VALS);
  // printf("Number of blocks: %d\n", BLOCKS);

  // Create caliper ConfigManager object
  cali::ConfigManager mgr;
  mgr.start();

  float *values = (float*) malloc( NUM_VALS * sizeof(float));
  CALI_MARK_BEGIN("data_init");
  array_fill(values, NUM_VALS);
  CALI_MARK_END("data_init");

  CALI_MARK_BEGIN("comp");
  CALI_MARK_BEGIN("comp_large");

  bitonic_sort(values); /* Inplace */

  CALI_MARK_END("comp");
  CALI_MARK_END("comp_large");

  CALI_MARK_BEGIN("correctness_check");
  int correctness = isSorted(values, NUM_VALS);
  CALI_MARK_END("correctness_check");

  adiak::init(NULL);
  adiak::launchdate();    // launch date of the job
  adiak::libraries();     // Libraries used
  adiak::cmdline();       // Command line used to launch the job
  adiak::clustername();   // Name of the cluster
  adiak::value("Algorithm", "bitonic_sort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
  adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
  adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
  adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
  adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
  adiak::value("InputType", options[OPTION-1]); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
  adiak::value("num_threads", THREADS); // The number of CUDA or OpenMP threads
  adiak::value("num_blocks", BLOCKS); // The number of CUDA blocks 
  adiak::value("group_num", 5); // The number of your group (integer, e.g., 1, 10)
  adiak::value("implementation_source", "Online and Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
  adiak::value("correctness", correctness); // Whether the dataset has been sorted (0, 1)

  // Flush Caliper output before finalizing MPI
  mgr.stop();
  mgr.flush();
}