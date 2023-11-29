/*
 * Parallel merger sort using CUDA.
 * Compile with
 * nvcc merge.cu
 */

#include "helper.cuh"

void recordAdiak(const char* dt, int dt_size, size_t len, int op, int threads, int blocks, bool sorted)
{
  const char* options[] = {"random", "sorted", "reverse_sorted", "1%perturbed"};
  adiak::init(NULL);
  adiak::launchdate();    // launch date of the job
  adiak::libraries();     // Libraries used
  adiak::cmdline();       // Command line used to launch the job
  adiak::clustername();   // Name of the cluster
  adiak::value("Algorithm", "merge_sort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
  adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
  adiak::value("Datatype", dt); // The datatype of input elements (e.g., double, int, float)
  adiak::value("SizeOfDatatype", dt_size); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
  adiak::value("InputSize", len); // The number of elements in input dataset (1000)
  adiak::value("InputType", options[op-1]); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
  adiak::value("num_threads", threads); // The number of CUDA or OpenMP threads
  adiak::value("num_blocks", blocks); // The number of CUDA blocks 
  adiak::value("group_num", 5); // The number of your group (integer, e.g., 1, 10)
  adiak::value("implementation_source", "Online and Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
  adiak::value("correctness", int(sorted)); // Whether the dataset has been sorted (0, 1)

}

// KERNEL FUNCTIONS

template<typename T>
__device__ inline
void Merge(T* values, T* result, int beg, int mid, int end)
{
  //        beg               mid             end
  // values [a . . . . . . . . b . . . . . . . .]
  // result [c . . . . . . . . . . . . . . . . .]
  int ai, bi, ri;
  ai = beg; bi = mid; ri = beg;
  while (ai<mid && bi<end) { 
    if (values[ai]<=values[bi]) {result[ri]=values[ai]; ai++;} 
    else {result[ri]=values[bi]; bi++;}
    ri++;
  }
  

  while (ai<mid) result[ri++]=values[ai++]; // ai++; ri++;
  while (bi<end) result[ri++]=values[bi++]; // bi++; ri++;

  for (int i=beg; i<end; i++) values[i]=result[i];
}

template<typename T>
__global__ static
void MergeSort(T* values, T* result, size_t len)
{
  extern __shared__ T shared[];

  const unsigned int tid = threadIdx.x;

  // Copy input to shared mem.
  shared[tid] = values[tid];
  
  __syncthreads();
  
  int u;
  for (int k = 1; k < len; k *= 2)
  {
    for (int i = 1; i+k <= len; i += k*2)
    {
      u = i + k*2;
      if(u > len) u = len + 1;
      Merge(shared, result, i, i+k, u);
    }
    __syncthreads();
  }
  
  values[tid] = shared[tid];
}


// Sorting Algorithm: Parallel Merge Sort
template<typename T>
milliseconds_t sort(T* values, T* result, size_t bytes, size_t len, int threads, int blocks)
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  
  cudaEventRecord(start);
  
  CALI_MARK_BEGIN("comp");
  CALI_MARK_BEGIN("comp_large");
  MergeSort<<<blocks, threads, bytes*2>>>(values, result, len);
  cudaDeviceSynchronize();
  CALI_MARK_END("comp_large");
  CALI_MARK_END("comp");

  cudaEventRecord(stop);

  cudaEventSynchronize(stop);

  milliseconds_t time = 0.0;
  cudaEventElapsedTime(&time, start, stop);
  return time;
}

template<typename T>
int run(size_t len, int threads, int option)
{
  char const* datatype;
  int blocks  = len / threads;
  printf("Number of threads: %d\n", threads);  // THREADS = 256; 
  printf("Number of values: %d\n", len);  // NUM_VALS = 1024;
  printf("Number of blocks: %d\n", blocks);
  printf("Array Type:\t%s\n", (datatype = (typeid(T) == typeid(int)) ? "int" : (typeid(T) == typeid(float)) ? "float" : "unknown"));

  clock_t start, stop;
  Timers cu_time;
  size_t bytes = len * sizeof(T);

  // Create caliper ConfigManager object
  cali::ConfigManager mgr; mgr.start();

  // Main Functionality
  start = clock();

  T* host_arr = alloc_arr<T>(len, option);                                     // 1. Allocate host memory and initialized host data
  T* D_values = alloc_device<T>(bytes);                                         // 2. Allocate device memory
  T* D_result = alloc_device<T>(bytes);
  cu_time.copyH2D = copy<T>(D_values, host_arr, bytes, cudaMemcpyHostToDevice); // 3. Transfer input data from host to device memory
  cu_time.sort = sort<T>(D_values, D_result, bytes, len, threads, blocks);      // 4. Execute kernels
  cu_time.copyD2H = copy<T>(host_arr, D_result, bytes, cudaMemcpyDeviceToHost); // 5. Transfer output from device memory to host

  stop = clock();

  // Data Calculation and Operations
  cu_time.bandwidth = bandwidth(cu_time.sort / 1000, bytes, 1);

  print_elapsed(start, stop);
  print_stats(&cu_time);

  bool sorted = check_dealloc<T>(host_arr, D_values, D_result, len);

  recordAdiak(datatype, sizeof(T), len, option, threads, blocks, sorted);
  
  // Flush Caliper output
  mgr.stop(); mgr.flush();

  return sorted - 1;    // <==> (sorted) ? 0 : -1
}

int main(int argc, char *argv[])
{
  CALI_CXX_MARK_FUNCTION;
  
  // Parse Args
  int threads = atoi(argv[1]);
  size_t length = atoi(argv[2]);
  int option = atoi(argv[3]);

  // Run algorithm on given args
  return run<float>(length, threads, option);
}