/*#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

// CUDA kernel for subtraction
__global__ void subtractionKernel(const float* input, float* a_output, float* b_output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n / 2; i += stride) {
        int j = i * 2;
        a_output[i] = input[j] - input[j + 2];
        b_output[i] = input[j + 1] - input[j + 3];
    }
}

// CUDA kernel for multiplication (with unrolling)
__global__ void multiplicationKernel(const float* a_input, const float* b_input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        output[i] = a_input[i] * b_input[i];
    }
}

__device__ float warp_reduce_sum(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

// CUDA kernel for summation (with shared memory and warp-level primitives)
__global__ void sumArrayKernel(const float* input, float* output, int n) {
    extern __shared__ float shared_data[];

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }

    // Warp-level reduction
    sum = warp_reduce_sum(sum);

    // Write warp-level sums to shared memory
    if (lane_id == 0) {
        shared_data[warp_id] = sum;
    }
    __syncthreads();

    // Parallel reduction in shared memory
    if (tid == 0) {
        sum = 0.0f;
        for (int i = 0; i < blockDim.x / 32; i++) {
            sum += shared_data[i];
        }
        output[blockIdx.x] = sum;
    }
}

int main() {
    const int N = 1000000; // Size of the input array

    float* input, * a_sub, * b_sub, * mult, * sum;
    float* d_input, * d_a_sub, * d_b_sub, * d_mult, * d_sum;

    // Allocate memory on the host
    input = new float[N];
    a_sub = new float[N / 2];
    b_sub = new float[N / 2];
    mult = new float[N / 2];
    sum = new float[1];

    // Initialize input data (omitted for brevity)
    // ...

    // Allocate memory on the device
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_a_sub, (N / 2) * sizeof(float));
    cudaMalloc(&d_b_sub, (N / 2) * sizeof(float));
    cudaMalloc(&d_mult, (N / 2) * sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));

    // Set up the grid and block dimensions
    const int blockSize = 256;
    const int numBlocks = (N / 2 + blockSize - 1) / blockSize;

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Copy input data from host to device
    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the subtraction kernel
    subtractionKernel << <numBlocks, blockSize >> > (d_input, d_a_sub, d_b_sub, N);
    cudaDeviceSynchronize();

    // Launch the multiplication kernel
    multiplicationKernel << <numBlocks, blockSize >> > (d_a_sub, d_b_sub, d_mult, N / 2);
    cudaDeviceSynchronize();

    // Launch the summation kernel
    sumArrayKernel << <numBlocks, blockSize, blockSize * sizeof(float) >> > (d_mult, d_sum, N / 2);
    cudaDeviceSynchronize();

    // Copy the result from device to host
    cudaMemcpy(sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();

    // Print the result
    std::cout << "Final result: " << sum[0] << std::endl;

    // Print the execution time
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Execution time: " << duration / 1000.0 << " ms" << std::endl;

    // Free memory
    delete[] input;
    delete[] a_sub;
    delete[] b_sub;
    delete[] mult;
    delete[] sum;
    cudaFree(d_input);
    cudaFree(d_a_sub);
    cudaFree(d_b_sub);
    cudaFree(d_mult);
    cudaFree(d_sum);

    return 0;
}*/


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <windows.h>
#include <time.h>


#include <stdio.h>
#include <stdlib.h>


#define BLOCK_SIZE 256

float** generateData(int n) {
    float** darray;

    // Allocate memory
    darray = (float**)malloc(sizeof(float*) * n + 1);
    if (darray == NULL) return NULL;

    int i;
    for (i = 0; i < n; i++)
    {
        darray[i] = (float*)malloc(sizeof(float));
    }

    float x;
    for (i = 0; i < n; i++)
    {
        x = rand(); // Generates a random number
        x = x / 10000;
        if (darray[i] != NULL) *(darray[i]) = x;
    }

    return darray;
}

float* generateData1(int n) {
    

    float* myheaparr = (float *)malloc(sizeof(float) * n);

    float x;
    int i;
    for (i = 0; i < n; i++)
    {
        x = rand(); // Generates a random number
        x = x / 10000;
        if (myheaparr[i] != NULL) (myheaparr[i]) = x;
    }

    return myheaparr;

}

// CUDA kernel to perform parallel reduction
__global__ void sumKernel(float* input, float* output, int n) {
    __shared__ float sharedMem[256];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load elements from global memory into shared memory
    float sum = 0;
    while (i < n) {
        sum += input[i];
        if (i + blockDim.x < n)
            sum += input[i + blockDim.x];
        i += blockDim.x * 2;
    }

    // Perform warp-level reduction
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Store the result to shared memory
    if (tid % warpSize == 0) {
        sharedMem[tid / warpSize] = sum;
    }

    // Ensure all threads within the block have completed writing to shared memory
    __syncthreads();

    // Perform block-level reduction
    int idx = threadIdx.x;
    if (tid < warpSize) {
        sum = (idx < blockDim.x / warpSize) ? sharedMem[idx] : 0;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }

    // Store the final result to global memory
    if (tid == 0) {
        output[blockIdx.x] = sum;
    }
}

__global__ void sumKernel1(float* input, float* output, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (int i = tid; i < n; i += stride) {
        if (i % 4 == 0) { // a_i
            float diff_a = input[i] - input[i + 2];
            float diff_b = input[i + 1] - input[i + 3];
            sum += diff_a * diff_b;
        }
    }

    // Store partial result to shared memory
    __shared__ float sharedSum[BLOCK_SIZE];
    sharedSum[threadIdx.x] = sum;
    __syncthreads();

    // Parallel reduction within the block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sharedSum[threadIdx.x] += sharedSum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Write block sum to global memory
    if (threadIdx.x == 0) {
        output[blockIdx.x] = sharedSum[0];
    }
}

// Function to launch CUDA kernel and perform summation
void sumArray(float* d_input, float* d_output, int n) {
    dim3 blockSize(256, 1, 1);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, 1, 1);

    sumKernel << <gridSize, blockSize >> > (d_input, d_output, n);
}

__global__ void summationKernel(const float* data, int size, float* result) {
    __shared__ float shared_data[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (i < size / 4) {
        shared_data[tid] = (data[4 * i] - data[4 * i + 2]) * (data[4 * i + 1] - data[4 * i + 3]);
    }
    else {
        shared_data[tid] = 0.0f;
    }
    __syncthreads();

    // Perform parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    // Write the result for each block to global memory
    if (tid == 0) {
        result[blockIdx.x] = shared_data[0];
    }
}

int main() {
    const int size = 1000000;
    //float data[size] = { 4.23, 7.85, 3.11, 6.42, 9.67, 2.34, 5.79, 8.16, 1.58, 4.95, 7.31, 2.69, 6.07, 9.44, 1.82, 5.25, 8.63, 3.99, 7.41, 2.14, 5.52, 8.89, 1.27, 4.65, 7.03, 2.39, 5.77, 9.15, 1.53, 4.92, 8.29, 3.67, 6.05, 9.83, 2.21, 5.64, 8.02, 1.44, 4.81, 7.19, 2.56, 5.93, 9.31, 1.69, 4.07, 7.45, 2.83, 6.21, 9.59, 3.97, 7.34, 1.72, 5.10, 8.48, 2.86, 6.24, 9.62, 3.00, 6.38, 9.76, 1.14, 4.52, 7.90, 2.28, 5.66, 8.04, 1.42, 4.80, 8.18, 3.56, 6.94, 2.32, 5.70, 9.08, 1.46, 4.84, 8.22, 3.60, 6.98, 2.36, 5.74, 9.12, 1.50, 4.88, 8.26, 3.64, 7.02, 2.40, 5.78, 9.16, 1.54, 4.92, 8.30, 3.68, 7.06, 2.44, 5.82, 9.20, 1.58, 4.96, 8.34, 3.72, 7.10, 2.48, 5.86, 9.24, 1.62, 5.00, 8.38, 3.76, 7.14, 2.52, 5.90, 9.28, 1.66, 5.04, 8.42, 3.80, 7.18, 2.56, 5.94, 9.32, 1.70, 5.08, 8.46, 3.84, 7.22, 2.60, 5.98, 9.36, 1.74, 5.12, 8.50, 3.88, 7.26, 2.64, 6.02, 9.40, 1.78, 5.16, 8.54, 3.92, 7.30, 2.68, 6.06, 9.44, 1.82, 5.20, 8.58, 3.96, 7.34, 2.72, 6.10, 9.48, 1.86, 5.24, 8.62, 4.00, 7.38, 2.76, 6.14, 9.52, 1.90, 5.28, 8.66, 4.04, 7.42, 2.80, 6.18, 9.56, 1.94, 5.32, 8.70, 4.08, 7.46, 2.84, 6.22, 9.60, 1.98, 5.36, 8.74, 4.12, 7.50, 2.88, 6.26, 9.64, 2.02, 5.40, 8.78, 4.16, 7.54, 2.92, 6.30, 9.68, 2.06, 5.44, 8.82, 4.20, 7.58, 2.96, 6.34, 9.72, 2.10, 5.48, 8.86, 4.24, 7.62, 3.00, 6.38, 9.76, 2.14, 5.52, 8.90, 4.28, 7.66, 3.04, 6.42, 9.80, 2.18, 5.56, 8.94, 4.32, 7.70, 3.08, 6.46, 9.84, 2.22, 5.60, 8.98, 4.36, 7.74, 3.12, 6.50, 9.88, 2.26, 5.64, 9.02, 4.40, 7.78, 3.16, 6.54, 9.92, 2.30, 5.68, 9.06, 4.44, 7.82, 3.20, 6.58, 9.96, 2.34, 5.72, 9.10, 4.48, 7.86, 3.54, 6.92, 4.29, 7.67, 1.05, 4.43, 7.81, 2.19, 5.57, 8.95, 3.33, 6.71, 1.09, 4.47, 7.85, 2.23, 5.61, 8.99, 3.37, 6.75, 1.13, 4.51, 7.89, 2.27, 5.65, 9.03, 3.41, 6.79, 1.17, 4.55, 7.93, 2.31, 5.69, 9.07, 3.45, 6.83, 1.21, 4.59, 7.97, 2.35, 5.73, 9.11, 3.49, 6.87, 1.25, 4.63, 8.01, 2.39, 5.77, 9.15, 3.53, 6.91, 1.29, 4.67, 8.05, 2.43, 5.81, 9.19, 3.57, 6.95, 1.33, 4.71, 8.09, 2.47, 5.85, 9.23, 3.61, 6.99, 1.37, 4.75, 8.13, 2.51, 5.89, 9.27, 3.65, 7.03, 1.41, 4.79, 8.17, 2.55, 5.93, 9.31, 3.69, 7.07, 1.45, 4.83, 8.21, 2.59, 5.97, 9.35, 3.73, 7.11, 1.49, 4.87, 8.25, 2.63, 6.01, 9.39, 3.77, 7.15, 1.53, 4.91, 8.29, 2.67, 6.05, 9.43, 3.81, 7.19, 1.57, 4.95, 8.33, 2.71, 6.09, 9.47, 3.85, 7.23, 1.61, 4.99, 8.37, 2.75, 6.13, 9.51, 3.89, 7.27, 1.65, 5.03, 8.41, 2.79, 6.17, 9.55, 3.93, 7.31, 1.69, 5.07, 8.45, 2.83, 6.21, 9.59, 3.97, 7.35, 1.73, 5.11, 8.49, 2.87, 6.25, 9.63, 4.01, 7.39, 1.77, 5.15, 8.53, 2.91, 6.29, 9.67, 4.05, 7.43, 1.81, 5.19, 8.57, 2.95, 6.33, 9.71, 4.09, 7.47, 1.85, 5.23, 8.61, 2.99, 6.37, 9.75, 4.13, 7.51, 1.89, 5.27, 8.65, 3.03, 6.41, 9.79, 4.17, 7.55, 1.93, 5.31, 8.69, 3.07, 6.45, 9.83, 4.21, 7.59, 1.97, 5.35, 8.73, 3.11, 6.49, 9.87, 4.25, 7.63, 2.01, 5.39, 8.77, 3.15, 6.53, 9.91, 4.29, 7.67, 2.05, 5.43, 8.81, 3.19, 6.57, 9.95, 4.33, 7.71, 2.09, 5.47, 8.85, 3.23, 6.61, 9.99, 4.37, 7.75, 2.13, 5.51, 8.89, 3.27, 6.65, 1.03, 4.41, 7.79, 2.17, 5.55, 8.93, 3.31, 6.69, 1.07, 4.45, 7.83, 2.21, 5.59, 8.97, 3.35, 6.73, 1.11, 4.49, 7.87, 2.25, 5.63, 9.01, 3.39, 6.77, 1.15, 4.53, 7.91, 2.29, 5.67, 9.05, 3.43, 6.81 };
    float* data = generateData1(size);
    float* d_data;
    float* d_result;
    float* h_result;
    int num_blocks = (size / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Calculate the time taken by fun() 
    clock_t t;
    t = clock();

    // Allocate device memory
    cudaMalloc((void**)&d_data, size * sizeof(float));
    cudaMalloc((void**)&d_result, num_blocks * sizeof(float));
    h_result = (float*)malloc(num_blocks * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_data, data, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    summationKernel << <num_blocks, BLOCK_SIZE >> > (d_data, size, d_result);

    // Copy the result from device to host
    cudaMemcpy(h_result, d_result, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);


    // Sum the partial results from each block
    float final_result = 0.0f;
    for (int i = 0; i < num_blocks; i++) {
        final_result += h_result[i];
    }

    t = clock() - t;
    double time_taken = ((double)t); // in seconds 
    printf("fun() took %f seconds to execute \n", time_taken);

    printf("Final result: %f\n", final_result);

    // Free device memory
    cudaFree(d_data);
    cudaFree(d_result);
    free(h_result);

    return 0;
}

int mainKarthik() {
    // Assuming you have initialized and filled inputArray with data

    time_t start, end;


    //int n = 1000000; // Size of the array
    int n = 512; // Size of the array
    size_t size = n * sizeof(float);

    //float** inputArray = generateData(n);

    //float** inputArray = { 4.23, 7.85, 3.11, 6.42, 9.67, 2.34, 5.79, 8.16, 1.58, 4.95, 7.31, 2.69, 6.07, 9.44, 1.82, 5.25, 8.63, 3.99, 7.41, 2.14, 5.52, 8.89, 1.27, 4.65, 7.03, 2.39, 5.77, 9.15, 1.53, 4.92, 8.29, 3.67, 6.05, 9.83, 2.21, 5.64, 8.02, 1.44, 4.81, 7.19, 2.56, 5.93, 9.31, 1.69, 4.07, 7.45, 2.83, 6.21, 9.59, 3.97, 7.34, 1.72, 5.10, 8.48, 2.86, 6.24, 9.62, 3.00, 6.38, 9.76, 1.14, 4.52, 7.90, 2.28, 5.66, 8.04, 1.42, 4.80, 8.18, 3.56, 6.94, 2.32, 5.70, 9.08, 1.46, 4.84, 8.22, 3.60, 6.98, 2.36, 5.74, 9.12, 1.50, 4.88, 8.26, 3.64, 7.02, 2.40, 5.78, 9.16, 1.54, 4.92, 8.30, 3.68, 7.06, 2.44, 5.82, 9.20, 1.58, 4.96, 8.34, 3.72, 7.10, 2.48, 5.86, 9.24, 1.62, 5.00, 8.38, 3.76, 7.14, 2.52, 5.90, 9.28, 1.66, 5.04, 8.42, 3.80, 7.18, 2.56, 5.94, 9.32, 1.70, 5.08, 8.46, 3.84, 7.22, 2.60, 5.98, 9.36, 1.74, 5.12, 8.50, 3.88, 7.26, 2.64, 6.02, 9.40, 1.78, 5.16, 8.54, 3.92, 7.30, 2.68, 6.06, 9.44, 1.82, 5.20, 8.58, 3.96, 7.34, 2.72, 6.10, 9.48, 1.86, 5.24, 8.62, 4.00, 7.38, 2.76, 6.14, 9.52, 1.90, 5.28, 8.66, 4.04, 7.42, 2.80, 6.18, 9.56, 1.94, 5.32, 8.70, 4.08, 7.46, 2.84, 6.22, 9.60, 1.98, 5.36, 8.74, 4.12, 7.50, 2.88, 6.26, 9.64, 2.02, 5.40, 8.78, 4.16, 7.54, 2.92, 6.30, 9.68, 2.06, 5.44, 8.82, 4.20, 7.58, 2.96, 6.34, 9.72, 2.10, 5.48, 8.86, 4.24, 7.62, 3.00, 6.38, 9.76, 2.14, 5.52, 8.90, 4.28, 7.66, 3.04, 6.42, 9.80, 2.18, 5.56, 8.94, 4.32, 7.70, 3.08, 6.46, 9.84, 2.22, 5.60, 8.98, 4.36, 7.74, 3.12, 6.50, 9.88, 2.26, 5.64, 9.02, 4.40, 7.78, 3.16, 6.54, 9.92, 2.30, 5.68, 9.06, 4.44, 7.82, 3.20, 6.58, 9.96, 2.34, 5.72, 9.10, 4.48, 7.86, 3.54, 6.92, 4.29, 7.67, 1.05, 4.43, 7.81, 2.19, 5.57, 8.95, 3.33, 6.71, 1.09, 4.47, 7.85, 2.23, 5.61, 8.99, 3.37, 6.75, 1.13, 4.51, 7.89, 2.27, 5.65, 9.03, 3.41, 6.79, 1.17, 4.55, 7.93, 2.31, 5.69, 9.07, 3.45, 6.83, 1.21, 4.59, 7.97, 2.35, 5.73, 9.11, 3.49, 6.87, 1.25, 4.63, 8.01, 2.39, 5.77, 9.15, 3.53, 6.91, 1.29, 4.67, 8.05, 2.43, 5.81, 9.19, 3.57, 6.95, 1.33, 4.71, 8.09, 2.47, 5.85, 9.23, 3.61, 6.99, 1.37, 4.75, 8.13, 2.51, 5.89, 9.27, 3.65, 7.03, 1.41, 4.79, 8.17, 2.55, 5.93, 9.31, 3.69, 7.07, 1.45, 4.83, 8.21, 2.59, 5.97, 9.35, 3.73, 7.11, 1.49, 4.87, 8.25, 2.63, 6.01, 9.39, 3.77, 7.15, 1.53, 4.91, 8.29, 2.67, 6.05, 9.43, 3.81, 7.19, 1.57, 4.95, 8.33, 2.71, 6.09, 9.47, 3.85, 7.23, 1.61, 4.99, 8.37, 2.75, 6.13, 9.51, 3.89, 7.27, 1.65, 5.03, 8.41, 2.79, 6.17, 9.55, 3.93, 7.31, 1.69, 5.07, 8.45, 2.83, 6.21, 9.59, 3.97, 7.35, 1.73, 5.11, 8.49, 2.87, 6.25, 9.63, 4.01, 7.39, 1.77, 5.15, 8.53, 2.91, 6.29, 9.67, 4.05, 7.43, 1.81, 5.19, 8.57, 2.95, 6.33, 9.71, 4.09, 7.47, 1.85, 5.23, 8.61, 2.99, 6.37, 9.75, 4.13, 7.51, 1.89, 5.27, 8.65, 3.03, 6.41, 9.79, 4.17, 7.55, 1.93, 5.31, 8.69, 3.07, 6.45, 9.83, 4.21, 7.59, 1.97, 5.35, 8.73, 3.11, 6.49, 9.87, 4.25, 7.63, 2.01, 5.39, 8.77, 3.15, 6.53, 9.91, 4.29, 7.67, 2.05, 5.43, 8.81, 3.19, 6.57, 9.95, 4.33, 7.71, 2.09, 5.47, 8.85, 3.23, 6.61, 9.99, 4.37, 7.75, 2.13, 5.51, 8.89, 3.27, 6.65, 1.03, 4.41, 7.79, 2.17, 5.55, 8.93, 3.31, 6.69, 1.07, 4.45, 7.83, 2.21, 5.59, 8.97, 3.35, 6.73, 1.11, 4.49, 7.87, 2.25, 5.63, 9.01, 3.39, 6.77, 1.15, 4.53, 7.91, 2.29, 5.67, 9.05, 3.43, 6.81 };

    float inputArray[] = {4.23, 7.85, 3.11, 6.42, 9.67, 2.34, 5.79, 8.16, 1.58, 4.95, 7.31, 2.69, 6.07, 9.44, 1.82, 5.25, 8.63, 3.99, 7.41, 2.14, 5.52, 8.89, 1.27, 4.65, 7.03, 2.39, 5.77, 9.15, 1.53, 4.92, 8.29, 3.67, 6.05, 9.83, 2.21, 5.64, 8.02, 1.44, 4.81, 7.19, 2.56, 5.93, 9.31, 1.69, 4.07, 7.45, 2.83, 6.21, 9.59, 3.97, 7.34, 1.72, 5.10, 8.48, 2.86, 6.24, 9.62, 3.00, 6.38, 9.76, 1.14, 4.52, 7.90, 2.28, 5.66, 8.04, 1.42, 4.80, 8.18, 3.56, 6.94, 2.32, 5.70, 9.08, 1.46, 4.84, 8.22, 3.60, 6.98, 2.36, 5.74, 9.12, 1.50, 4.88, 8.26, 3.64, 7.02, 2.40, 5.78, 9.16, 1.54, 4.92, 8.30, 3.68, 7.06, 2.44, 5.82, 9.20, 1.58, 4.96, 8.34, 3.72, 7.10, 2.48, 5.86, 9.24, 1.62, 5.00, 8.38, 3.76, 7.14, 2.52, 5.90, 9.28, 1.66, 5.04, 8.42, 3.80, 7.18, 2.56, 5.94, 9.32, 1.70, 5.08, 8.46, 3.84, 7.22, 2.60, 5.98, 9.36, 1.74, 5.12, 8.50, 3.88, 7.26, 2.64, 6.02, 9.40, 1.78, 5.16, 8.54, 3.92, 7.30, 2.68, 6.06, 9.44, 1.82, 5.20, 8.58, 3.96, 7.34, 2.72, 6.10, 9.48, 1.86, 5.24, 8.62, 4.00, 7.38, 2.76, 6.14, 9.52, 1.90, 5.28, 8.66, 4.04, 7.42, 2.80, 6.18, 9.56, 1.94, 5.32, 8.70, 4.08, 7.46, 2.84, 6.22, 9.60, 1.98, 5.36, 8.74, 4.12, 7.50, 2.88, 6.26, 9.64, 2.02, 5.40, 8.78, 4.16, 7.54, 2.92, 6.30, 9.68, 2.06, 5.44, 8.82, 4.20, 7.58, 2.96, 6.34, 9.72, 2.10, 5.48, 8.86, 4.24, 7.62, 3.00, 6.38, 9.76, 2.14, 5.52, 8.90, 4.28, 7.66, 3.04, 6.42, 9.80, 2.18, 5.56, 8.94, 4.32, 7.70, 3.08, 6.46, 9.84, 2.22, 5.60, 8.98, 4.36, 7.74, 3.12, 6.50, 9.88, 2.26, 5.64, 9.02, 4.40, 7.78, 3.16, 6.54, 9.92, 2.30, 5.68, 9.06, 4.44, 7.82, 3.20, 6.58, 9.96, 2.34, 5.72, 9.10, 4.48, 7.86, 3.54, 6.92, 4.29, 7.67, 1.05, 4.43, 7.81, 2.19, 5.57, 8.95, 3.33, 6.71, 1.09, 4.47, 7.85, 2.23, 5.61, 8.99, 3.37, 6.75, 1.13, 4.51, 7.89, 2.27, 5.65, 9.03, 3.41, 6.79, 1.17, 4.55, 7.93, 2.31, 5.69, 9.07, 3.45, 6.83, 1.21, 4.59, 7.97, 2.35, 5.73, 9.11, 3.49, 6.87, 1.25, 4.63, 8.01, 2.39, 5.77, 9.15, 3.53, 6.91, 1.29, 4.67, 8.05, 2.43, 5.81, 9.19, 3.57, 6.95, 1.33, 4.71, 8.09, 2.47, 5.85, 9.23, 3.61, 6.99, 1.37, 4.75, 8.13, 2.51, 5.89, 9.27, 3.65, 7.03, 1.41, 4.79, 8.17, 2.55, 5.93, 9.31, 3.69, 7.07, 1.45, 4.83, 8.21, 2.59, 5.97, 9.35, 3.73, 7.11, 1.49, 4.87, 8.25, 2.63, 6.01, 9.39, 3.77, 7.15, 1.53, 4.91, 8.29, 2.67, 6.05, 9.43, 3.81, 7.19, 1.57, 4.95, 8.33, 2.71, 6.09, 9.47, 3.85, 7.23, 1.61, 4.99, 8.37, 2.75, 6.13, 9.51, 3.89, 7.27, 1.65, 5.03, 8.41, 2.79, 6.17, 9.55, 3.93, 7.31, 1.69, 5.07, 8.45, 2.83, 6.21, 9.59, 3.97, 7.35, 1.73, 5.11, 8.49, 2.87, 6.25, 9.63, 4.01, 7.39, 1.77, 5.15, 8.53, 2.91, 6.29, 9.67, 4.05, 7.43, 1.81, 5.19, 8.57, 2.95, 6.33, 9.71, 4.09, 7.47, 1.85, 5.23, 8.61, 2.99, 6.37, 9.75, 4.13, 7.51, 1.89, 5.27, 8.65, 3.03, 6.41, 9.79, 4.17, 7.55, 1.93, 5.31, 8.69, 3.07, 6.45, 9.83, 4.21, 7.59, 1.97, 5.35, 8.73, 3.11, 6.49, 9.87, 4.25, 7.63, 2.01, 5.39, 8.77, 3.15, 6.53, 9.91, 4.29, 7.67, 2.05, 5.43, 8.81, 3.19, 6.57, 9.95, 4.33, 7.71, 2.09, 5.47, 8.85, 3.23, 6.61, 9.99, 4.37, 7.75, 2.13, 5.51, 8.89, 3.27, 6.65, 1.03, 4.41, 7.79, 2.17, 5.55, 8.93, 3.31, 6.69, 1.07, 4.45, 7.83, 2.21, 5.59, 8.97, 3.35, 6.73, 1.11, 4.49, 7.87, 2.25, 5.63, 9.01, 3.39, 6.77, 1.15, 4.53, 7.91, 2.29, 5.67, 9.05, 3.43, 6.81};

    

    // Calculate the time taken by fun() 
    clock_t t;
    t = clock();
   
    

    //time(&start);
    //ios_base::sync_with_stdio(false);

    // Allocate memory on the GPU
    float* d_input, * d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_input, inputArray, size, cudaMemcpyHostToDevice);


    /*int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (int i = tid; i < n; i += stride) {
        if (i % 2 == 0) { // a_i
            float diff_a = d_input[i] - d_input[i + 1];
            float diff_b = d_input[i + 2] - d_input[i + 3];
            sum += diff_a * diff_b;
        }
    }*/


    // Launch kernel
    sumKernel1 <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >>> (d_input, d_output, n);

    //sumArray(d_input, d_output, n);

    // Copy the result back to host
    float result = 0.0;
    cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    t = clock() - t;
    double time_taken = ((double)t) ; // in seconds 
    //time(&end);

    //double time_taken = double(end - start);

    printf("fun() took %f seconds to execute \n", time_taken);
    // Print result
    //printf("Sum: %f\n", result);

    //char str[256];
    //sprintf_s(str, sizeof(str), "It works! - number: %f \n", result);


    //OutputDebugStringA(str);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

int main2() {
    // Generate array of float numbers
    int data_size = 1000000;
    //int data_size = 8;
    float** darray = generateData(data_size);
    if (darray == NULL) return 0;

    int array_size = sizeof(darray);
    // Maximum size of array
    size_t size = array_size * sizeof(float);

    // Allocate memory for the source and destination on the GPU
    float* src;
    cudaMalloc(&src, size);
    float* dest;
    // Calculate total blocks of threads
    int gridSize = data_size / 256;     // Block size
    cudaMalloc(&dest, gridSize * sizeof(float));

    // Copy data from host to device    
    cudaMemcpy(src, darray, size, cudaMemcpyHostToDevice);

    // Launch kernel
    //sumArray(src, dest, data_size);
    // Launch kernel
    sumKernel1 << <(data_size + 256 - 1) / 256, 256 >> > (src, dest, data_size);


    // Copy the result back to host
    float result = 0;
    cudaMemcpy(&result, dest, sizeof(float), cudaMemcpyDeviceToHost);

    //float result1 = (0.409999996 - 63.3400002) * (184.669998 - 265.000000) * (191.690002 - 114.779999) * (157.240005 - 293.579987);
    //printf("Sum: %f\n", result1);

    // Print result
    printf("Sum: %f\n", result);

    // Free memory
    cudaFree(src);
    cudaFree(dest);

    return 0;
}

/*float** digitarray_gen(int ele_count) {

}*/





cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

__global__ void addKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main1()
{
    // ADC Digitizer Array
    float digit_array[] = { 3.544, 5.544, 6.544, 6.544 };
    int array_size = sizeof(digit_array);

    // Maximum size of array
    size_t size = sizeof(digit_array) * sizeof(float);

    // GU memory allocation
    float* input, * output;
    cudaMalloc(&input, size);
    cudaMalloc(&output, size);



    // Allocate memory on the GPU
    float* d_input, * d_output;

    // cudaMalloc(&d_output, gridSize.x * sizeof(float));

    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    //addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}
