#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstring>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>


/*
a list of sources:
- (main one) https://www.geeksforgeeks.org/implementation-of-quick-sort-using-mpi-omp-and-posix-thread/#
- https://www.codeproject.com/KB/threads/Parallel_Quicksort/Parallel_Quick_sort_without_merge.pdf
*/

int NUM_VALS;
int OPTION;

const char* options[4] = {"sorted", "reverse_sorted", "random", "1%perturbed"};

float random_float() {
  return (float)rand()/(float)RAND_MAX;
}

void array_fill(float *arr, int length, int option, int rank, int numtasks) {
    if (option == 1) { 
        int start = length * rank;
        for (int i = 0; i < length; ++i) {
            arr[i] = (float)(start + i);
        }
    } else if (option == 2) {
        int start = length * (numtasks - rank - 1);
        for (int i = 0; i < length; ++i) {
            arr[i] = (float)(start - i);
        }
    } else if (option == 3) {
        srand(MPI_Wtime() * 1000 + rank);
        for (int i = 0; i < length; ++i) {
            arr[i] = random_float();
        }
    } else if (option == 4) {
        int start = length * rank;
        for (int i = 0; i < length; ++i) {
            arr[i] = (float)(start + i);
        }
        srand(MPI_Wtime() * 1000 + rank);
        int perturb_count = length / 100;
        for (int i = 0; i < perturb_count; ++i) {
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


void quicksort(float* values, int left, int right) {
    if (right <= 1) {
        return;
    }

    float pivot = values[left + right / 2];
    swap(&values[left], &values[left + right / 2]);

    int index = left;

    for (int i = left + 1; i < left + right; i++) {
        if (values[i] < pivot) {
            index++;
            swap(&values[i], &values[index]);
        }
    }

    swap(&values[left], &values[index]);

    quicksort(values, left, index - left);
    quicksort(values, index + 1, left + right - index - 1);
}

float* merge_sorted(float* arr1, int n1, float* arr2, int n2) {
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

    int numtasks, taskid, chunk_size, local_chunk_size;
    float *data = NULL, *chunk = NULL, *sorted = NULL;

    NUM_VALS = atoi(argv[1]);
    OPTION = atoi(argv[2]);

    MPI_Status status;

    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        fprintf(stderr, "MPI Initialization failed.\n");
        return -1;
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    cali::ConfigManager mgr;
    mgr.start();

    data = (float*)malloc(NUM_VALS * sizeof(float));
    CALI_MARK_BEGIN("data_init");
    array_fill(data, NUM_VALS, OPTION, taskid, numtasks);
    CALI_MARK_END("data_init");

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("MPI_Barrier");
    MPI_Barrier(MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Barrier");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("MPI_Bcast");
    MPI_Bcast(&NUM_VALS, 1, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Bcast");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    chunk_size = (NUM_VALS % numtasks == 0) ? (NUM_VALS / numtasks) : (NUM_VALS / (numtasks - 1));
    chunk = (float*)malloc(chunk_size * sizeof(float));

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("MPI_Scatter");
    MPI_Scatter(data, chunk_size, MPI_FLOAT, chunk, chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Scatter");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    free(data);
    
    // float *data = NULL;
    local_chunk_size = (NUM_VALS >= chunk_size * (taskid + 1)) ? chunk_size : (NUM_VALS - chunk_size * taskid);
    
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    quicksort(chunk, 0, local_chunk_size);
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");
    
    for (int step = 1; step < numtasks; step *= 2) {
        if (taskid % (2 * step) != 0) {
            CALI_MARK_BEGIN("comm");
            CALI_MARK_BEGIN("comm_large");
            CALI_MARK_BEGIN("MPI_Send");
            MPI_Send(chunk, local_chunk_size, MPI_FLOAT, taskid - step, 0, MPI_COMM_WORLD);
            CALI_MARK_END("MPI_Send");
            CALI_MARK_END("comm_large");
            CALI_MARK_END("comm");
            break;
        }

        if (taskid + step < numtasks) {
            int received_chunk_size = (NUM_VALS >= chunk_size * (taskid + 2 * step)) ? (chunk_size * step) : (NUM_VALS - chunk_size * (taskid + step));

            float* received_chunk = (float*)malloc(received_chunk_size * sizeof(float));

            CALI_MARK_BEGIN("comm");
            CALI_MARK_BEGIN("comm_large");
            CALI_MARK_BEGIN("MPI_Recv");
            MPI_Recv(received_chunk, received_chunk_size, MPI_FLOAT, taskid + step, 0, MPI_COMM_WORLD, &status);
            CALI_MARK_END("MPI_Recv");
            CALI_MARK_END("comm_large");
            CALI_MARK_END("comm");

            CALI_MARK_BEGIN("comp");
            CALI_MARK_BEGIN("comp_large");
            sorted = merge_sorted(chunk, local_chunk_size, received_chunk, received_chunk_size);
            CALI_MARK_END("comp_large");
            CALI_MARK_END("comp");

            free(chunk);
            free(received_chunk);
            chunk = sorted;
            local_chunk_size += received_chunk_size;
        }
    }

    sorted = (float*)malloc(NUM_VALS * sizeof(float));  

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("MPI_Gather");
    MPI_Gather(chunk, local_chunk_size, MPI_FLOAT, sorted, local_chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Gather");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    if (taskid == 0) {
        CALI_MARK_BEGIN("correctness_check");
        int correctness = check(sorted, NUM_VALS);
        CALI_MARK_END("correctness_check");

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

        // free(sorted);
    }
    
    mgr.stop();
    mgr.flush();

    MPI_Finalize();

    free(sorted);
}