#include <stdio.h>
#include <time.h>
#include <algorithm>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>


// TYPES
typedef float seconds_t;
typedef float milliseconds_t;
typedef float GBperSecond_t;
typedef struct {
  GBperSecond_t bandwidth = 0.0;
  milliseconds_t sort = 0.0;
  milliseconds_t copyH2D = 0.0;
  milliseconds_t copyD2H = 0.0;
} Timers;


// HELPER FUNCTIONS

int random(int) { return (int)rand() % (int)INT_MAX; }

float random(float) { return (float)rand() / (float)RAND_MAX; }

inline void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double)(stop - start)) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.3fs\n", elapsed);
}

template<typename T>
inline void print_array(T* arr, size_t length, const char* title = "")
{
  printf("%s \n{", title);
  for (int i = 0; i < length; ++i) printf(" %d,", arr[i]);
  printf("\b}\n");
}

inline void print_stats(Timers* ct)
{
  printf("Sorting Step Time:\t\t %.6f ms\n", ct->sort);
  printf("cudaMemcpy Host2Dev Time:\t\t %.6f ms\n", ct->copyH2D);
  printf("cudaMemcpy Dev2Host Time:\t\t %.6f ms\n", ct->copyD2H);
  printf("Effective Bandwidth (GB/s):\t\t %.6f GB/s\n", ct->bandwidth);
}

inline GBperSecond_t bandwidth(seconds_t time, size_t bytes, int num_calls)
{ // EFFECTIVE BANDWIDTH
  #define RW_OPS 4
  return (RW_OPS * num_calls * bytes * 1e-9) / time;
}

template<typename T>
inline void fill_array(T* arr, size_t len, int option)
{ 
  CALI_MARK_BEGIN("data_init");
  srand(0);
  switch (option)
  {
    case 1: // Random
      for (int i = 0; i < len; ++i) arr[i] = random(T());    break;
    case 2: // Sorted
      for (int i = 0; i < len; ++i) arr[i] = T(i);           break;
    case 3: // Reverse Sorted
      for (int i = 0; i < len; ++i) arr[i] = T(len-1-i);  break;
    case 4: // 1% Perturbed
      for (int i = 0; i < len; ++i) arr[i] = T(i);
      for (int i = 0; i < len / 100; ++i) arr[rand() % len] = random(T());
      break;
    default: break;
  }
  CALI_MARK_END("data_init");
}

template<typename T>
inline bool check_dealloc(T* __host, T* __dev1, T* __dev2, size_t len)
{
  // print_array(__host, len, "\nAfter Sort:");
  CALI_MARK_BEGIN("correctness_check");
  bool pass = std::is_sorted(&__host[0], &__host[len - 1]);
  CALI_MARK_END("correctness_check");

  cudaFree(__dev1);
  cudaFree(__dev2);
  delete[] __host;

  if (pass) printf("\t\033[1m\033[92mSORTED\033[0m\n");
  else printf("\t\033[1m\033[31mNOT SORTED\033[0m\n");
  return pass;
}


// MAJOR STEPS

template<typename T>
T* alloc_host(int len, int option)
{
  T* values = new T[len];
  fill_array<T>(values, len, option);
  return values;
}

template<typename T>
T* alloc_device(size_t bytes)
{
  T* dev_values;
  cudaMalloc((void **)&dev_values, bytes);
  return dev_values;
}

template<typename T>
milliseconds_t copy(T* dst, T* src, size_t bytes, cudaMemcpyKind direction)
{
  // MEM COPY FROM HOST TO DEVICEs
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  
  cudaEventRecord(start);
  CALI_MARK_BEGIN("comm");
  CALI_MARK_BEGIN("comm_large");
  CALI_MARK_BEGIN("cudaMemcpy");
  cudaError_t err = cudaMemcpy(dst, src, bytes, direction);
  CALI_MARK_END("cudaMemcpy");
  CALI_MARK_END("comm_large");
  CALI_MARK_END("comm");
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);

  if (err) printf("%s\n", cudaGetErrorString(err));

  milliseconds_t time = 0.0;
  cudaEventElapsedTime(&time, start, stop);
  return time;
}