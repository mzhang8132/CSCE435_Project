#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#include <cuda_runtime.h>
#include <cuda.h>

int THREADS;
int BLOCKS;
int NUM_VALS;
int OPTION;

const char* options[3];
options[0] = "random";
options[1] = "sorted";
options[2] = "reverse_sorted";

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
        for (int i = length-1; i >= 0; --i) {
            arr[i] = (float)i;
        }
    }
}

void swap(float *A, float *B) {
    float temp = *A;
    *A = *B;
    *B = temp;
}

bool check(float *values, int num_vals) {
    for (int i = 1; i < num_vals; ++i){
        if (values[i] < values[i-1]) {
            return false
        }
    }
    return true
}

__global__ void odd_even_sort(float *values, int num_vals) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_vals) {
        for (int phase = 0; phase < num_vals; phase++) {
            if (phase % 2 == 0) {  // Even phase
                if (i % 2 == 0 & i < num_vals - 1) {
                    if (values[i] > values[i + 1]) {
                        swap(values[i], values[i+1]);
                    }
                }
            } else {  // Odd phase
                if (i % 2 != 0 & i < num_vals - 1) {
                    if (values[i] > values[i + 1]) {
                        swap(values[i], values[i+1]);
                    }
                }
            }
        }
    }
}

__global__ void check_sorted(float *values, int num_vals, int *is_sorted) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int local_idx = threadIdx.x;

    extern __shared__ float local_data[];

    // Load data into shared memory
    if (i < num_vals) {
        local_data[local_idx] = values[i];
    }

    // Synchronize to ensure all data is loaded
    __syncthreads();

    if (local_idx < blockDim.x - 1) {
        if (local_data[local_idx] > local_data[local_idx + 1]) {
            *is_sorted = 0;
        }
    }
}

int main(int argc, char *argv[]) {
    CALI_CXX_MARK_FUNCTION;

    THREADS = atoi(argv[1]);
    NUM_VALS = atoi(argv[2]);
    OPTION = atoi(argv[3]);
    BLOCKS = NUM_VALS / THREADS;

    size_t size = NUM_VALS * sizeof(float);

    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();

    // Generate values
    float *values = (float*) malloc( NUM_VALS * sizeof(float));
    CALI_MARK_BEGIN("data_init");
    array_fill(values, NUM_VALS, OPTION);
    CALI_MARK_END("data_init");

    float *dev_values;
    cudaMalloc((void**) &dev_values, size);

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN(cudaMemcpy_host_to_device);
    cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);
    CALI_MARK_END(cudaMemcpy_host_to_device);
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");


    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    for (int i = 0; i < NUM_VALS; i++) {
        odd_even_sort<<<BLOCKS, THREADS>>>(dev_values, NUM_VALS);
    }
    cudaDeviceSynchronize();
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    int dev_is_sorted = 1;

    CALI_MARK_BEGIN("correctness_check");
    check_sorted<<<BLOCKS, THREADS, THREADS * sizeof(float)>>>(dev_values, NUM_VALS, dev_is_sorted);
    cudaDeviceSynchronize();
    CALI_MARK_END("correctness_check");

    if (dev_is_sorted == 1) {
        printf("Array is sorted");
    } else {
        printf("Array is not sorted");
    }

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "odd_even_sort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
    adiak::value("InputType", options[OPTION-1]); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_threads", THREADS); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", BLOCKS); // The number of CUDA blocks 
    adiak::value("group_num", 5); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online and Handwritten") // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    // Flush Caliper output
    mgr.stop();
    mgr.flush();
}