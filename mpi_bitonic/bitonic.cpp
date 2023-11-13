//ADAPTED FROM https://cs.indstate.edu/~jkinne/cs670-s2013/code/sort/mergeBitonicMPI.cpp
/*
Evan Burriola
    Original only supported integer inputs, rewrote Data gen, MPI functions,
    comparators, and verifier to support float inputs

    Data generation supports global random, global sequential, global reversed.

*/

//------------------------------------------------------------------------
//AUTHOR COMMENT
/*
  compile: g++ mergeBitonic.cpp
  run: ./a.out n
  note: n will be rounded up to the next higher power of 2.

  This code implements bitonic sort.  For details, see 
  http://en.wikipedia.org/wiki/Bitonic_sorter


  This is the MPI version.  Basically we took the "swap" from the 
  main loop and converted it to compareExchange.

  Note that this may be slower than mergeSort, etc.  But it is a 
  sorting network - the same exact comparisons are done regardless
  of what the actual lists of numbers are.

  And we haven't really optimized.  To make it faster, should do something
  smarter instead of the qsort's.  Because each half is already sorted, 
  so should just need to merge them.

  Note also - doesn't work for 1 node.
 */
//------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "mpi.h"

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

int NUM_VALS;
int OPTION;
int numTasks, rank;

const char* options[3] = {"random", "sorted", "reverse_sorted"};

float random_float() {
  return (float)rand()/(float)RAND_MAX;
}

float *array_fill(int length, int offset, int option) {
    float * numbers = (float *) malloc(sizeof(float) * length);
    if (option == 1) {
        srand(offset);
        for (int i = 0; i < length; ++i) {
            numbers[i] = random_float();
        }
    } else if (option == 2) {
        for (int i = 0; i < length; ++i) {
            numbers[i] = (float)i + offset;
        }
    } else if (option == 3) {
        for (int i = 0; i < length; ++i) {
            numbers[i] = (float)length-1-i + offset;
        }
    }

    return numbers;
}

// print array of howMany numbers.
void printNumbers(float * numbers, int howMany) {
  printf("\n");
  for(int i=0; i < howMany; i++)
    printf("%1.3f\n", numbers[i]);
  printf("\n");
}

// check if array of howMany random numbers is sorted in increasing order.
// return 1 or 0.
int isSorted(float *numbers, int howMany) {
  for(int i=1; i<howMany; i++) {
    if (numbers[i] < numbers[i-1]) return 0;
  }
  return 1;
}

//Ref: https://www.geeksforgeeks.org/comparator-function-of-qsort-in-c/
int compareDescending(const void *item1, const void *item2) {
  float x = * ( (float *) item1), y = * ( (float *) item2);
  return y-x < 0 ? -1 : y-x == 0 ? 0 : 1;
}

int compareAscending(const void *item1, const void *item2) {
  float x = * ( (float *) item1), y = * ( (float *) item2);
  return x-y < 0 ? -1 : x-y == 0 ? 0 : 1;
}

float *tempArray;

void compareExchange(float *numbers, int howMany, 
		     int node1, int node2, int biggerFirst,
		     int sequenceNo) {
  if (node1 != rank && node2 != rank) return;

  memcpy(tempArray, numbers, howMany*sizeof(float));

  MPI_Status status;

  // get numbers from the other node. 
  // have the process that is node1 always send first, and node2 
  // receive first - they can't both send at the same time.
  int nodeFrom = node1==rank ? node2 : node1;
  if (node1 == rank) {
    MPI_Send(numbers, howMany, MPI_FLOAT, nodeFrom, sequenceNo, MPI_COMM_WORLD);
    MPI_Recv(&tempArray[howMany], howMany, MPI_FLOAT, nodeFrom, sequenceNo, 
	     MPI_COMM_WORLD, &status);
  }
  else {
    MPI_Recv(&tempArray[howMany], howMany, MPI_FLOAT, nodeFrom, sequenceNo, 
	     MPI_COMM_WORLD, &status);
    MPI_Send(numbers, howMany, MPI_FLOAT, nodeFrom, sequenceNo, MPI_COMM_WORLD);
  }

  // sort them.
  if (biggerFirst) {
    qsort(tempArray, howMany*2, sizeof(float), compareDescending);
  }
  else {
    qsort(tempArray, howMany*2, sizeof(float), compareAscending);
  }

  // keep only half of them.
  if (node1 == rank)
    memcpy(numbers, tempArray, howMany*sizeof(float));
  else
    memcpy(numbers, &tempArray[howMany], howMany*sizeof(float));
}

/*
  function: mergeBitonic - perform bitonic merge sort.
*/
void mergeBitonic(float *numbers, int howMany) {
  tempArray = (float *) malloc(sizeof(float) * howMany * 2);

  int log = numTasks;
  int pow2i = 2;
  int sequenceNumber = 0;

  for(int i=1; log > 1 ; i++) {
    int pow2j = pow2i;
    for(int j=i; j >= 1; j--) {
      sequenceNumber++;
      for(int node=0; node < numTasks; node += pow2j) {
	for(int k=0; k < pow2j/2; k++) {
	  //printf("i=%d, j=%d, node=%d, k=%d, pow2i=%d, pow2j=%d\n", 
	  // i, j, node, k, pow2i, pow2j);
	  compareExchange(numbers, howMany, node+k, node+k+pow2j/2, 
			  ((node+k) % (pow2i*2) >= pow2i),
			  sequenceNumber);
	}
      }
      pow2j /= 2;

        //    printf(" after substage %d", j);
        //    printNumbers(numbers, howMany);
    }
    pow2i *= 2;
    log /= 2;

    //    printf("after stage %d\n", i);
    //    printNumbers(numbers, howMany);
  }

  free(tempArray);
}

int main(int argc, char *argv[]) {
  CALI_CXX_MARK_FUNCTION;
  int howMany;
  long int returnVal;
  int len;
  char hostname[MPI_MAX_PROCESSOR_NAME];

  // initialize
  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &numTasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Get_processor_name(hostname, &len);

  cali::ConfigManager mgr;
  mgr.start();

  NUM_VALS = atoi(argv[1]);
  OPTION = atoi(argv[2]);
  howMany = NUM_VALS/numTasks;

  int offset = rank*howMany;
  // each process creates a list of random numbers.
  CALI_MARK_BEGIN("data_init");
  float * numbers = array_fill(howMany, offset, OPTION);
  CALI_MARK_END("data_init");
//   printNumbers(numbers, howMany);

  CALI_MARK_BEGIN("comp");
  CALI_MARK_BEGIN("comp_large");
  mergeBitonic(numbers, howMany);
  CALI_MARK_END("comp_large");
  CALI_MARK_END("comp");
  
//   printNumbers(numbers, howMany);

  // they are all sorted, now just gather them up.
  float * allNumbers = NULL;
  if (rank == 0) {
    allNumbers = (float *) malloc(howMany * numTasks * sizeof(float));
  }

  CALI_MARK_BEGIN("comm");
  CALI_MARK_BEGIN("comm_large");
  CALI_MARK_BEGIN("MPI_Gather");
  //Gather all values into the global array
  MPI_Gather(numbers, howMany, MPI_FLOAT, 
	     allNumbers, howMany, MPI_FLOAT, 
	     0, MPI_COMM_WORLD);
  CALI_MARK_END("MPI_Gather");
  CALI_MARK_END("comm_large");
  CALI_MARK_END("comm");
  

  //Verify!
  if (rank == 0) {
    CALI_MARK_BEGIN("correctness_check");
    int correct = isSorted(allNumbers, howMany * numTasks);
    // if (correct)
    //   printf("Successfully sorted!\n");
    // else
    //   printf("Error: numbers not sorted.\n");
    CALI_MARK_END("correctness_check");
    
    // printNumbers(allNumbers, howMany * numTasks);

    free(allNumbers);

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "bitonic_sort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
    adiak::value("InputType", options[OPTION-1]); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", numTasks); // The number of processors (MPI ranks)
    adiak::value("group_num", 5); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
    adiak::value("correctness", correct); // Whether the dataset has been sorted (0, 1)

  }
  
  free(numbers);

  // Flush Caliper output before finalizing MPI
  mgr.stop();
  mgr.flush();

  MPI_Finalize();
  return 0;
}