#include "mpi.h"
#include <algorithm>


#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

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
T* sort(int height, int id, T localArray[], size_t len, MPI_Comm comm, T globalArray[])
{
  int parent, rightChild, myHeight;
  T *half1, *half2, *mergeResult;

  myHeight = 0;
  std::sort(&localArray[0], &localArray[len]); // sort small, local array using sequential means
  half1 = localArray;

  while (myHeight < height) {  
    parent = (id & (~(1 << myHeight)));

    if (parent == id) {
      rightChild = (id | (1 << myHeight));


      half2 = new T[len];
      // MPI_Recv(half2, len, MPI_INT, rightChild, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


      mergeResult = new T[len*2];

      mergeResult = merge<T>(half1, half2, mergeResult, len);

      half1 = mergeResult;
      len = len * 2;

      delete[] half2;
      mergeResult = NULL;

      myHeight++;

    } else {
        
      // MPI_Send(half1, len, MPI_INT, parent, 0, MPI_COMM_WORLD);
      if (myHeight != 0) delete[] half1;
      myHeight = height;
    }
  }

  if (id == 0) globalArray = half1;
  return globalArray;
}


template<typename T>
int run(int procs, size_t len, int height, int option)
{

  int id, local_len;
  T* localArray;
  T* globalArray;
  double local_time, total_time, master_time, process_time;
  
  int length = -1;
  char myHostName[MPI_MAX_PROCESSOR_NAME];
  MPI_Comm_rank(MPI_COMM_WORLD, &id);           // Get the task ID
  MPI_Get_processor_name(myHostName, &length);

  if (id == 0) {
    globalArray = new T[len];
    fill_array(globalArray, len, option);
    // print_array(globalArray, len, "UNSORTED ARRAY", id);
  }

  local_len = len / procs;
  localArray = new T[local_len];
  MPI_Scatter(globalArray, local_len, MPI_INT, localArray, local_len, MPI_INT, 0, MPI_COMM_WORLD);

  // print_array(localArray, local_len, "localArray", id);

  // local_time = MPI_Wtime();
  if (id == 0) {
    // master_time = MPI_Wtime();
    globalArray = sort<T>(height, id, localArray, local_len, MPI_COMM_WORLD, globalArray);
    // master_time = MPI_Wtime() - master_time;
    printf("Process #%d of %d on %s took %f seconds \n", id, procs, myHostName, master_time);
  } else {
    // process_time = MPI_Wtime();
    sort<T>(height, id, localArray, local_len, MPI_COMM_WORLD, NULL);
    // process_time = MPI_Wtime() - process_time;
    printf("Process #%d of %d on %s took %f seconds \n", id, procs, myHostName, process_time);
  }
  std::cout << "Get local_time\tPROC:" << id << "\n";
  // local_time = MPI_Wtime() - local_time;
  std::cout << "MPI_Reduce\t\tPROC:" << id << "\n";
  // MPI_Reduce(&local_time, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (id == 0) {
    // print_array(globalArray, len, "FINAL SORTED ARRAY", id);
    printf("Sorting %d integers took %f seconds \n", len, total_time);
    bool sorted = std::is_sorted(&globalArray[0], &globalArray[len]);
    if (sorted) std::cout << "SORTED\n";
    else {
      std::cout << "NOT SORTED\n";
      print_array(globalArray, len, "FAILED SORTED ARRAY");
    }
    delete[] globalArray;
  }

  delete[] localArray;
  std::cout << "delete[] localArray\tPROC:" << id << "\n";
  return id;
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  
  // Parse Args
  size_t length = atoi(argv[1]);
  int procs, height, option = atoi(argv[2]);

  MPI_Comm_size(MPI_COMM_WORLD, &procs);     // Get Number of Processors

  // calculate total height of tree
  height = __builtin_ctz(procs);   // Equivalent to logBASE2


  // Run algorithm on given args
  int id = run<int>(procs, length, height, option);
  std::cout << "Finished run()\tPROC:" << id << "\n";

  // std::cout << "Howdy world" << std::endl;

  // MPI_Waitall(); <-- MAYBE?
  MPI_Finalize();
  return 0;
}