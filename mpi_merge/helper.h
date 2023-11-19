#include <iostream>
#include <algorithm>

#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

// HELPER FUNCTIONS

int random(int) { return (int)rand() % (int)10; }

float random(float) { return (float)rand() / (float)RAND_MAX; }

template<typename T>
inline void print_array(T* arr, size_t length, const char* title = "", int id = -1)
{
  std::cout << title;
  if (id != -1) std::cout << "\tID:" << id;
  std::cout << "\n{";
  for (int i = 0; i < length; ++i) std::cout << " " << arr[i] << ",";
  std::cout << "\b}\n";
}

template<typename T>
inline void fill_array(T* arr, size_t len, int option)
{ 
  // std::cout << "Filling array with option:" << option << "\n";
  CALI_MARK_BEGIN("data_init");
  srand(0);
  switch (option)
  {
    case 1: // Random
      for (int i = 0; i < len; ++i) arr[i] = random(T()); break;
    case 2: // Sorted
      for (int i = 0; i < len; ++i) arr[i] = T(i);        break;
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
inline bool check_dealloc(T* __arr, size_t len)
{
  // print_array(__arr, len, "\nAfter Sort:");
  CALI_MARK_BEGIN("correctness_check");
  bool pass = std::is_sorted(&__arr[0], &__arr[len]);
  CALI_MARK_END("correctness_check");

  delete[] __arr;

  // if (pass) printf("\t\033[1m\033[92mSORTED\033[0m\n");
  // else printf("\t\033[1m\033[31mNOT SORTED\033[0m\n");
  return pass;
}

// MAJOR STEPS

template<typename T>
T* alloc_arr(int len, int option)
{
  T* values = new T[len];
  fill_array<T>(values, len, option);
  return values;
}


void recordAdiak(const char* dt, int dt_size, size_t len, int op, int procs, bool sorted)
{
  const char* options[] = {"random", "sorted", "reverse_sorted", "1%perturbed"};
  adiak::init(NULL);
  adiak::launchdate();    // launch date of the job
  adiak::libraries();     // Libraries used
  adiak::cmdline();       // Command line used to launch the job
  adiak::clustername();   // Name of the cluster
  adiak::value("Algorithm", "merge_sort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
  adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
  adiak::value("Datatype", dt); // The datatype of input elements (e.g., double, int, float)
  adiak::value("SizeOfDatatype", dt_size); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
  adiak::value("InputSize", len); // The number of elements in input dataset (1000)
  adiak::value("InputType", options[op-1]); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
  adiak::value("num_procs", procs); // The number of CUDA or OpenMP threads
  adiak::value("group_num", 5); // The number of your group (integer, e.g., 1, 10)
  adiak::value("implementation_source", "Online and Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
  adiak::value("correctness", int(sorted)); // Whether the dataset has been sorted (0, 1)
}