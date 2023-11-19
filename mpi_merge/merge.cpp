#include "mpi.h"

#include "helper.h"


/*  Work in progress
a list of sources:
- (main one) https://selkie-macalester.org/csinparallel/modules/MPIProgramming/build/html/mergeSort/mergeSort.html
- https://github.com/racorretjer/Parallel-Merge-Sort-with-MPI/blob/master/merge-mpi.c
- https://stackoverflow.com/questions/61910607/parallel-merge-sort-using-mpi
*/

/**
 * @brief Merge two sublists into one Result list
 * @param A     Sublist to merge
 * @param B     Sublist to merge
 * @param res   Resulting list of merged elements
 * @param size  Size of both sublists
*/
template<typename T>
T* merge(T* A, T* B, T* res, size_t size)
{
  int ai, bi, ri;
  ai = bi = ri = 0;
  while (ai < size && bi < size)
  { 
    if (A[ai] <= B[bi]) res[ri++] = A[ai++];
    else res[ri++] = B[bi++];
  }
  
  // Copy remainder of list
  while (ai < size) res[ri++] = A[ai++];
  while (bi < size) res[ri++] = B[bi++];

  return res;
}

template<typename T>
T* mergeSort(int height, int id, T localArray[], size_t len, T globalArray[])
{
  int parent, rightChild, myHeight = 0;
  T* half1;
  T* half2;
  T* mergeResult;


  CALI_MARK_BEGIN("comp");
  CALI_MARK_BEGIN("comp_large");
  std::sort(&localArray[0], &localArray[len]); // sort small, local array using sequential means
  CALI_MARK_END("comp_large");
  CALI_MARK_END("comp");

  half1 = localArray;

  while (myHeight < height)
  {  
    parent = (id & (~(1 << myHeight)));
    if (parent == id) {
      rightChild = (id | (1 << myHeight));
      half2 = new T[len];
      mergeResult = new T[len*2];

      CALI_MARK_BEGIN("comm");
      CALI_MARK_BEGIN("comm_large");
      CALI_MARK_BEGIN("MPI_Recv");
      MPI_Recv(half2, len, MPI_INT, rightChild, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      CALI_MARK_END("MPI_Recv");
      CALI_MARK_END("comm_large");
      CALI_MARK_END("comm");


      CALI_MARK_BEGIN("comp");
      CALI_MARK_BEGIN("comp_large");
      mergeResult = merge<T>(half1, half2, mergeResult, len);
      CALI_MARK_END("comp_large");
      CALI_MARK_END("comp");

      half1 = mergeResult;
      len = len * 2;

      delete[] half2;
      mergeResult = NULL;

      myHeight++;

    } else {
      CALI_MARK_BEGIN("comm");
      CALI_MARK_BEGIN("comm_large");
      CALI_MARK_BEGIN("MPI_Send");
      MPI_Send(half1, len, MPI_INT, parent, 0, MPI_COMM_WORLD);
      CALI_MARK_END("MPI_Send");
      CALI_MARK_END("comm_large");
      CALI_MARK_END("comm");
      if (myHeight != 0) delete[] half1;
      myHeight = height;
    }
  }
  // printf("Process #%d Finished \n", id);
  if (id == 0) globalArray = half1;
  return globalArray;
}


template<typename T>
T* sort(T* values, size_t len, int procs, int height, int id)
{
  int local_len = len / procs;
  T* local_arr = new T[local_len];

  CALI_MARK_BEGIN("comm");
  CALI_MARK_BEGIN("comm_large");
  CALI_MARK_BEGIN("MPI_Scatter");
  MPI_Scatter(values, local_len, MPI_INT, local_arr, local_len, MPI_INT, 0, MPI_COMM_WORLD);
  CALI_MARK_END("MPI_Scatter");
  CALI_MARK_END("comm_large");
  CALI_MARK_END("comm");

  values = mergeSort<T>(height, id, local_arr, local_len, (id == 0) ? values : NULL);
  
  delete[] local_arr;
  return values;
}

template<typename T>
int run(size_t len, int procs, int option)
{
  int id; MPI_Comm_rank(MPI_COMM_WORLD, &id);

  char const* datatype = (typeid(T) == typeid(int)) ? "int" : (typeid(T) == typeid(float)) ? "float" : "unknown";
  int height = __builtin_ctz(procs);
  
  // Create caliper ConfigManager object
  cali::ConfigManager mgr; mgr.start();

  T* globl_arr;
  bool sorted = false;
  // MPI_Bcast(&length, 1, MPI_LONG, 0, MPI_COMM_WORLD); // Broadcast length to all procs

  if (id == 0) globl_arr = alloc_arr<T>(len, option);

  globl_arr = sort<T>(globl_arr, len, procs, height, id);
  
  if (id == 0) {
    sorted = check_dealloc<T>(globl_arr, len);
    recordAdiak(datatype, sizeof(T), len, option, procs, sorted);
  }

  // Flush Caliper output
  mgr.stop(); mgr.flush();

  return sorted - 1;
}

int main(int argc, char *argv[])
{
  CALI_CXX_MARK_FUNCTION;
  MPI_Init(&argc, &argv);
  
  // Parse Args
  size_t length = atoi(argv[1]);
  int option =    atoi(argv[2]);
  int procs;      MPI_Comm_size(MPI_COMM_WORLD, &procs);
  
  // Run algorithm on given args
  int res = run<float>(length, procs, option);

  MPI_Finalize();
  return res;
}