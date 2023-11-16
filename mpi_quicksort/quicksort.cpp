#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstring>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>


/*
a list of sources:
- (main one) https://github.com/triasamo1/Quicksort-Parallel-MPI/blob/master/quicksort_mpi.c
https://www.geeksforgeeks.org/implementation-of-quick-sort-using-mpi-omp-and-posix-thread/#
- https://www.codeproject.com/KB/threads/Parallel_Quicksort/Parallel_Quick_sort_without_merge.pdf
*/

int NUM_VALS;
int OPTION;

const char* options[4] = {"random", "sorted", "reverse_sorted", "1%perturbed"};

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
    } else if (option == 4) {
        for (int i = 0; i < length; ++i) {
            arr[i] = (float)offset+i;
        }
        
        int perturb_count = length / 100;
        srand(0);
        for (int i = 0; i < perturb_count; i++){
            int index = rand() % length;
            arr[index] = random_float();
        }
    }
}

void swap(float *x, float *y) {
  float temp = *x;
  *x = *y;
  *y = temp;
}


int partition(float *values, int left, int right) {
    int i = left, j = right;
    float pivot = values[(left + right) / 2];

    while (i <= j) {
        while (values[i] < pivot)
            i++;
        while (values[j] > pivot)
            j--;
        if (i <= j) {
            swap(&values[i], &values[j]);
            i++;
            j--;
        }
    }
    return j;
}

void quicksort(float *values, int left, int right) {
    if (left < right) {
        int pivot_index = partition(values, left, right);
        quicksort(values, left, pivot_index);
        quicksort(values, pivot_index + 1, right);
    }
}

float* merge(float* arr1, int n1, float* arr2, int n2) {
    float* result = (float*)malloc((n1 + n2) * sizeof(float));
    int i = 0, j = 0, k = 0;
    while (i < n1 && j < n2)
        result[k++] = arr1[i] < arr2[j] ? arr1[i++] : arr2[j++];
    while (i < n1)
        result[k++] = arr1[i++];
    while (j < n2)
        result[k++] = arr2[j++];
    return result;
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


int main(int argc, char *argv[]) {
    CALI_CXX_MARK_FUNCTION;

    int numtasks, taskid;

    NUM_VALS = atoi(argv[1]);
    OPTION = atoi(argv[2]);

    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    cali::ConfigManager mgr;
    mgr.start();

    int offset = NUM_VALS / numtasks;

    float *values = (float*)malloc(offset * sizeof(float));

    // Generate values
    CALI_MARK_BEGIN("data_init");
    array_fill(values, offset, offset * taskid, OPTION); 
    CALI_MARK_END("data_init");


    MPI_Barrier(MPI_COMM_WORLD);
    double start_timer = MPI_Wtime();

    CALI_MARK_BEGIN("comp");
    quicksort(values, 0, offset - 1);
    CALI_MARK_END("comp");
    
    for (int step = 1; step < numtasks; step = 2 * step) {
        int partner = taskid ^ step;
    
        if (partner < numtasks) {
            MPI_Status status;
            int partner_chunk_size;
    
            CALI_MARK_BEGIN("comm");
            MPI_Sendrecv(&offset, 1, MPI_INT, partner, 1,
                         &partner_chunk_size, 1, MPI_INT, partner, 1,
                         MPI_COMM_WORLD, &status);
            
            float* partner_chunk = (float*)malloc(partner_chunk_size * sizeof(float));
            MPI_Sendrecv(values, offset, MPI_FLOAT, partner, 2,
                         partner_chunk, partner_chunk_size, MPI_FLOAT, partner, 2,
                         MPI_COMM_WORLD, &status);
            CALI_MARK_END("comm");
            
            CALI_MARK_BEGIN("merge");
            float* merged = merge(values, offset, partner_chunk, partner_chunk_size);
            CALI_MARK_END("merge");
    
            free(values);
            free(partner_chunk);
            values = merged;
            offset += partner_chunk_size;
        }
    }
    
    float *global_list;

    CALI_MARK_BEGIN("comm");
    if (taskid == 0) {
        global_list = (float*)malloc(NUM_VALS * sizeof(float));
        assign(global_list, values, offset, 0);
        float *temp = (float*)malloc(offset * sizeof(float));
        for (int rank = 1; rank < numtasks; rank++) {
            MPI_Recv(temp, offset, MPI_FLOAT, rank, 0, MPI_COMM_WORLD, &status);
            assign(global_list, temp, offset, rank);
        }
        free(temp);
    } else {
        CALI_MARK_BEGIN("MPI_Send");
        MPI_Send(values, offset, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        CALI_MARK_END("MPI_Send");
    }
    CALI_MARK_END("comm");

    if (taskid == 0) {
        CALI_MARK_BEGIN("correctness_check");
        int correctness = check(global_list, NUM_VALS);
        CALI_MARK_END("correctness_check");
        
        /*
        for (int i = 0; i < NUM_VALS; i++){
        printf("%f ", global_list[i]);
        }
        printf("\n");
        */
        
        double finish_timer = MPI_Wtime();
        printf("Total time: %2.0f seconds\n", finish_timer - start_timer);

        adiak::init(NULL);
        adiak::launchdate();    // launch date of the job
        adiak::libraries();     // Libraries used
        adiak::cmdline();       // Command line used to launch the job
        adiak::clustername();   // Name of the cluster
        adiak::value("Algorithm", "quicksort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
        adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
        adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
        adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
        adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
        adiak::value("InputType", options[OPTION-1]); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
        adiak::value("num_procs", numtasks); // The number of processors (MPI ranks)
        adiak::value("group_num", 5); // The number of your group (integer, e.g., 1, 10)
        adiak::value("implementation_source", "Online, Handwritten, and AI"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
        adiak::value("correctness", correctness); // Whether the dataset has been sorted (0, 1)

        free(global_list);
    }

    free(values);

    mgr.stop();
    mgr.flush();

    MPI_Finalize();
}