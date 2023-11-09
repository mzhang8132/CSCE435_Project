#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstring>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

int NUM_VALS;
int OPTION;

const char* options[3] = {"random", "sorted", "reverse_sorted"};

float random_float() {
  return (float)rand()/(float)RAND_MAX;
}

void array_fill(float *arr, int length, int offset, int option) {
    if (option == 1) {
        srand(offset);
        for (int i = 0; i < length; ++i) {
            arr[i] = random_float();
        }
    } else if (option == 2) {
        for (int i = 0; i < length; ++i) {
            arr[i] = (float)offset+i;
        }
    } else if (option == 3) {
        for (int i = 0; i < length; ++i) {
            arr[i] = (float)offset+length-1-i;
        }
    }
}

int Compare(const void* a, const void* b) {
    return ( *(int*)a - *(int*)b );
}

void merge_low(float *values, float *A, float *B, int num_vals) {
   int a, b, c;
   
   a = 0;
   b = 0;
   c = 0;

   while (c < num_vals) {
      if (values[a] <= A[b]) {
         B[c] = values[a];
         c++; a++;
      } else {
         B[c] = A[b];
         c++; b++;
      }
   }

   memcpy(values, B, num_vals*sizeof(float));
}

void merge_high(float *values, float *A, float *B, int num_vals) {
   int a, b, c;
   
   a = num_vals - 1;
   b = num_vals - 1;
   c = num_vals - 1;

   while (c >= 0) {
      if (values[a] >= A[b]) {
         B[c] = values[a];
         c--; a--;
      } else {
         B[c] = A[b];
         c--; b--;
      }
   }

   memcpy(values, B, num_vals*sizeof(float));
}

void odd_even_iter(float *values, float *A, float *B, int num_vals, int phase, int even_partner, int odd_partner, int rank, int num_threads, MPI_Comm comm) {
   MPI_Status status;

   if (phase % 2 == 0) {
      if (even_partner >= 0) {
         MPI_Sendrecv(values, num_vals, MPI_FLOAT, even_partner, 0, A, num_vals, MPI_FLOAT, even_partner, 0, comm, &status);
         if (rank % 2 != 0)
            merge_high(values, A, B, num_vals);
         else
            merge_low(values, A, B, num_vals);
      }
   } else {
      if (odd_partner >= 0) {
         MPI_Sendrecv(values, num_vals, MPI_FLOAT, odd_partner, 0, A, num_vals, MPI_FLOAT, odd_partner, 0, comm, &status);
         if (rank % 2 != 0)
            merge_low(values, A, B, num_vals);
         else
            merge_high(values, A, B, num_vals);
      }
   }
}

void odd_even_sort(float *values, int num_vals, int rank, int num_threads, MPI_Comm comm) {
    int even_partner;
    int odd_partner;

    float *A = (float*) malloc(num_vals*sizeof(float));
    float *B = (float*) malloc(num_vals*sizeof(float));

    if (rank % 2 != 0) {   /* odd rank */
        even_partner = rank - 1;
        odd_partner = rank + 1;
        if (odd_partner == num_threads) odd_partner = MPI_PROC_NULL;  // Idle during odd phase
    } else {                   /* even rank */
        even_partner = rank + 1;
        if (even_partner == num_threads) even_partner = MPI_PROC_NULL;  // Idle during even phase
        odd_partner = rank-1;
    }

    
    qsort(values, num_vals, sizeof(int), Compare);

    for (int phase = 0; phase < num_threads; phase++) {
        odd_even_iter(values, A, B, num_vals, phase, even_partner, odd_partner, rank, num_threads, comm);
    }
    
    // deallocate memory
    free(A);
    free(B);
}

void assign(float* values, float* arr, int offset, int rank) {
    for (int i = 0; i < offset; ++i) {
        values[i+offset*rank] = arr[i];
    }
}

int check(float* values, int length) {
    for (int i = 1; i < length; ++i) {
        if (values[i] < values[i -1]) {
            return 0;
        }
    }
    return 1;
}

int main (int argc, char *argv[]) {
    CALI_CXX_MARK_FUNCTION;

    int	numtasks, taskid;
    
    NUM_VALS = atoi(argv[1]);
    OPTION = atoi(argv[2]);

    MPI_Status status;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
    MPI_Comm_size(MPI_COMM_WORLD,&numtasks);

    cali::ConfigManager mgr;
    mgr.start();

    int offset = NUM_VALS / numtasks;

    float *values = (float*) malloc(offset * sizeof(float));

    // Generate values
    CALI_MARK_BEGIN("data_init");
    array_fill(values, offset, offset*taskid, OPTION);
    CALI_MARK_END("data_init");

    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    odd_even_sort(values, offset, taskid, numtasks, MPI_COMM_WORLD);
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    float* global_list;

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    if (taskid == 0) {
        global_list = (float*) malloc( NUM_VALS * sizeof(float));
        assign(global_list, values, offset, 0);
        float *temp = (float*) malloc(offset * sizeof(float));
        for (int rank = 1; rank < numtasks; rank++) {
            MPI_Recv(temp, offset, MPI_FLOAT, rank, 0, MPI_COMM_WORLD, &status);
            assign(global_list, temp, offset, rank);
        }
        free(temp);
    } else {
        MPI_Send(values, offset, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    if (taskid == 0) {
        CALI_MARK_BEGIN("correctness_check");
        int correctness = check(global_list, NUM_VALS);
        CALI_MARK_END("correctness_check");
    
        adiak::init(NULL);
        adiak::launchdate();    // launch date of the job
        adiak::libraries();     // Libraries used
        adiak::cmdline();       // Command line used to launch the job
        adiak::clustername();   // Name of the cluster
        adiak::value("Algorithm", "odd_even_sort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
        adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
        adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
        adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
        adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
        adiak::value("InputType", options[OPTION-1]); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
        adiak::value("num_procs", numtasks); // The number of processors (MPI ranks)
        adiak::value("group_num", 5); // The number of your group (integer, e.g., 1, 10)
        adiak::value("implementation_source", "Online and Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
        adiak::value("correctness", correctness); // Whether the dataset has been sorted (0, 1)
    }

    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();

    MPI_Finalize();
}