#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

// source: https://github.com/saigowri/CUDA/blob/master/quicksort.cu
// source: https://medium.com/swlh/revisiting-quicksort-with-julia-and-cuda-2a997447939b 
// source: https://forums.developer.nvidia.com/t/quick-sort-depth/35100
// soucre: https://codepal.ai/code-generator/query/ek4N3nrB/function-in-c-iterative-quicksort-with-cuda

#include <cuda_runtime.h>
#include <cuda.h>

int THREADS;
int BLOCKS;
int NUM_VALS;
int OPTION;

const char* options[3] = {"random", "sorted", "reverse_sorted"};

float random_float() {
  return (float)rand()/(float)RAND_MAX;
}

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
        for (int i = 0; i < length; ++i) {
            arr[i] = (float)length-1-i;
        }
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

__device__ void swap(float *x, float *y) {
  float temp = *x;
  *x = *y;
  *y = temp;
}

__device__ int partition(float *values, int left, int right) {
    float pivot = values[(left + right) / 2];
    int i = left;
    int j = right;
    while (i <= j) {
        while (values[i] < pivot) i++;
        while (values[j] > pivot) j--;
        if (i <= j) {
            swap(&values[i], &values[j]);
            i++;
            j--;
        }
    }
    return i;
}


__global__ void quicksort(float *values, int length) {
    #define MAX_LEVELS 300

    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index >= length) return;

    int pivotIndex, L, R;
    int leftStack[MAX_LEVELS];
    int rightStack[MAX_LEVELS];

    int stackP = 0;
    leftStack[stackP] = 0;
    rightStack[stackP] = length - 1;

    while (stackP >= 0) {
        L = leftStack[stackP];
        R = rightStack[stackP];
        stackP--;

        if (L < R) {
            pivotIndex = partition(values, L, R);

            // Pushing right side to stack
            if (pivotIndex < R) {
                stackP++;
                leftStack[stackP] = pivotIndex;
                rightStack[stackP] = R;
            }

            // Pushing left side to stack
            if (L < pivotIndex - 1) {
                stackP++;
                leftStack[stackP] = L;
                rightStack[stackP] = pivotIndex - 1;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    CALI_CXX_MARK_FUNCTION;

    THREADS = atoi(argv[1]);
    NUM_VALS = atoi(argv[2]);
    OPTION = atoi(argv[3]);
    BLOCKS = (NUM_VALS + THREADS - 1) / THREADS;

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
    cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    // for (int i = 0; i < NUM_VALS; i++) {
    //     quicksort<<<BLOCKS, THREADS>>>(dev_values, i, NUM_VALS);
    // }
    quicksort<<<BLOCKS, THREADS>>>(dev_values, NUM_VALS);
    cudaDeviceSynchronize();
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    CALI_MARK_BEGIN("correctness_check");
    int correctness = check(values, NUM_VALS);
    CALI_MARK_END("correctness_check");

    cudaFree(dev_values);

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "quicksort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
    adiak::value("InputType", options[OPTION-1]); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_threads", THREADS); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", BLOCKS); // The number of CUDA blocks 
    adiak::value("group_num", 5); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online and Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
    adiak::value("correctness", correctness); // Whether the dataset has been sorted (0, 1)

    // Flush Caliper output
    mgr.stop();
    mgr.flush();
}