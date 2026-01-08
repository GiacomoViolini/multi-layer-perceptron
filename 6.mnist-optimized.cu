#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "utils.c"
#define INPUT_SIZE 784
#define OUTPUT_SIZE 10
#define N_NEURONS 256
#define ITERATIONS 200
#define LEARNING_RATE 0.10
#define N_TRAIN_SAMPLES 41000
#define N_TEST_SAMPLES 1000
#define CUDA_CHECK(call)                                                     \
    do                                                                       \
    {                                                                        \
        cudaError_t error = call;                                            \
        if (error != cudaSuccess)                                            \
        {                                                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error));                              \
            cudaDeviceReset();                                               \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

typedef struct
{
    float *d_W1, *d_W2, *d_b1, *d_b2;
    float *d_X, *d_Z1, *d_A1, *d_Z2, *d_A2;
    float *d_dW1, *d_dW2, *d_db1, *d_db2;
    float *d_dZ1, *d_dZ2, *d_dReLU;
    float *d_Y_one_hot;
    int *d_Y;
} GPUMemory;

__global__ void relu_kernel(float *Z, float *A, int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size)
        A[idx] = fmaxf(Z[idx], 0.0);
}

__global__ void softmax_kernel(float *Z, float *A, int samples)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= samples)
        return;
    float sum = 0.0;
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        float exp_val = expf(Z[i * samples + col]);
        A[i * samples + col] = exp_val;
        sum += exp_val;
    }

    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        A[i * samples + col] /= sum;
    }
}

__global__ void naive_matmul_kernel(float *A, float *B, float *C, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < k)
    {
        float sum = 0.0;
        for (int l = 0; l < n; l++)
        {
            sum += A[row * n + l] * B[l * k + col];
        }
        C[row * k + col] = sum;
    }
}

__global__ void naive_matmul_a_bt_kernel(float *A, float *B, float *C, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < k)
    {
        float sum = 0.0;
        for (int l = 0; l < n; l++)
        {
            sum += A[row * n + l] * B[col * n + l];
        }
        C[row * k + col] = sum;
    }
}

__global__ void naive_matmul_at_b_kernel(float *A, float *B, float *C, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < k)
    {
        float sum = 0.0;
        for (int l = 0; l < m; l++)
        {
            sum += A[l * n + row] * B[l * k + col];
        }
        C[row * k + col] = sum;
    }
}

__global__ void add_bias_kernel(float *Z, float *b, int neurons, int samples)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < neurons && col < samples)
    {
        Z[row * samples + col] += b[row];
    }
}

__global__ void one_hot_kernel(int *Y, int samples, float *Y_one_hot)
{
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample < samples)
    {
        int label = Y[sample];
        Y_one_hot[label * samples + sample] = 1.0;
    }
}

__global__ void compute_dZ2_kernel(float *A2, float *Y_one_hot, float *dZ2, int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size)
    {
        dZ2[idx] = A2[idx] - Y_one_hot[idx];
    }
}

__global__ void scale_kernel(float *A, float scale, int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size)
    {
        A[idx] *= scale;
    }
}

__global__ void avg_rows_kernel(float *input, float *output, int rows, int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows)
    {
        float sum = 0.0;
        for (int col = 0; col < cols; col++)
        {
            sum += input[row * cols + col];
        }
        output[row] = sum / cols;
    }
}

__global__ void relu_derivative_kernel(float *Z, float *dReLU, int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size)
        dReLU[idx] = Z[idx] > 0 ? 1.0 : 0.0;
}

__global__ void computedZ1(float *A, float *B, int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size)
    {
        A[idx] *= B[idx];
    }
}

__global__ void update_params_kernel(float *W, float *dW, float lr, int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size)
    {
        W[idx] -= lr * dW[idx];
    }
}

void allocate_gpu_memory(GPUMemory *gpu, int samples)
{
    CUDA_CHECK(cudaMalloc(&gpu->d_W1, N_NEURONS * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu->d_W2, OUTPUT_SIZE * N_NEURONS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu->d_b1, N_NEURONS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu->d_b2, OUTPUT_SIZE * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&gpu->d_X, samples * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu->d_Y, samples * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu->d_Z1, N_NEURONS * samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu->d_A1, N_NEURONS * samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu->d_Z2, OUTPUT_SIZE * samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu->d_A2, OUTPUT_SIZE * samples * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&gpu->d_dW1, N_NEURONS * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu->d_dW2, OUTPUT_SIZE * N_NEURONS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu->d_db1, N_NEURONS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu->d_db2, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu->d_dZ1, N_NEURONS * samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu->d_dZ2, OUTPUT_SIZE * samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu->d_dReLU, N_NEURONS * samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu->d_Y_one_hot, OUTPUT_SIZE * samples * sizeof(float)));
}

void free_gpu_memory(GPUMemory *gpu)
{
    CUDA_CHECK(cudaFree(gpu->d_W1));
    CUDA_CHECK(cudaFree(gpu->d_W2));
    CUDA_CHECK(cudaFree(gpu->d_b1));
    CUDA_CHECK(cudaFree(gpu->d_b2));
    CUDA_CHECK(cudaFree(gpu->d_X));
    CUDA_CHECK(cudaFree(gpu->d_Y));
    CUDA_CHECK(cudaFree(gpu->d_Z1));
    CUDA_CHECK(cudaFree(gpu->d_A1));
    CUDA_CHECK(cudaFree(gpu->d_Z2));
    CUDA_CHECK(cudaFree(gpu->d_A2));
    CUDA_CHECK(cudaFree(gpu->d_dW1));
    CUDA_CHECK(cudaFree(gpu->d_dW2));
    CUDA_CHECK(cudaFree(gpu->d_db1));
    CUDA_CHECK(cudaFree(gpu->d_db2));
    CUDA_CHECK(cudaFree(gpu->d_dZ1));
    CUDA_CHECK(cudaFree(gpu->d_dZ2));
    CUDA_CHECK(cudaFree(gpu->d_dReLU));
    CUDA_CHECK(cudaFree(gpu->d_Y_one_hot));
}

void forward_prop_gpu(GPUMemory *gpu, int samples)
{
    dim3 threadsPerBlockFirstLayer(32, 32);
    dim3 numBlocksFirstLayer((samples + threadsPerBlockFirstLayer.x - 1) / threadsPerBlockFirstLayer.x,
                             (N_NEURONS + threadsPerBlockFirstLayer.y - 1) / threadsPerBlockFirstLayer.y);
    naive_matmul_kernel<<<numBlocksFirstLayer, threadsPerBlockFirstLayer>>>(gpu->d_W1, gpu->d_X, gpu->d_Z1, N_NEURONS, INPUT_SIZE, samples);

    int total_hidden = samples * N_NEURONS;
    int grid_hidden = (total_hidden + 255) / 256;
    add_bias_kernel<<<grid_hidden, 256>>>(gpu->d_Z1, gpu->d_b1, N_NEURONS, samples);

    int ReLuBlocks = (N_NEURONS * samples + 256 - 1) / 256;
    relu_kernel<<<ReLuBlocks, 256>>>(gpu->d_Z1, gpu->d_A1, N_NEURONS * samples);

    dim3 threadsPerBlockSecondLayer(16, 16);
    dim3 numBlocksSecondLayer((samples + threadsPerBlockSecondLayer.x - 1) / threadsPerBlockSecondLayer.x,
                              (OUTPUT_SIZE + threadsPerBlockSecondLayer.y - 1) / threadsPerBlockSecondLayer.y);
    naive_matmul_kernel<<<numBlocksSecondLayer, threadsPerBlockSecondLayer>>>(gpu->d_W2, gpu->d_A1, gpu->d_Z2, OUTPUT_SIZE, N_NEURONS, samples);

    int total_out = samples * OUTPUT_SIZE;
    int grid_out = (total_out + 255) / 256;
    add_bias_kernel<<<grid_out, 256>>>(gpu->d_Z2, gpu->d_b2, OUTPUT_SIZE, samples);

    int softmaxBlocks = (samples + 31) / 32;
    softmax_kernel<<<softmaxBlocks, 32>>>(gpu->d_Z2, gpu->d_A2, samples);
}

void backward_prop_gpu(GPUMemory *gpu, int samples)
{
    CUDA_CHECK(cudaMemset(gpu->d_Y_one_hot, 0, OUTPUT_SIZE * samples * sizeof(float)));

    int diffblocks = (samples + 64 - 1) / 64;
    one_hot_kernel<<<diffblocks, 64>>>(gpu->d_Y, samples, gpu->d_Y_one_hot);

    int dZ2blocks = (samples * OUTPUT_SIZE + 64 - 1) / 64;
    compute_dZ2_kernel<<<dZ2blocks, 64>>>(gpu->d_A2, gpu->d_Y_one_hot, gpu->d_dZ2, OUTPUT_SIZE * samples);

    dim3 dW2blocks((N_NEURONS + 15) / 16, (OUTPUT_SIZE + 15) / 16);
    naive_matmul_a_bt_kernel<<<dW2blocks, dim3(16, 16)>>>(gpu->d_dZ2, gpu->d_A1, gpu->d_dW2, OUTPUT_SIZE, samples, N_NEURONS);

    int scaleW2blocks = (OUTPUT_SIZE * N_NEURONS + 64 - 1) / 64;
    scale_kernel<<<scaleW2blocks, 64>>>(gpu->d_dW2, 1.0 / samples, OUTPUT_SIZE * N_NEURONS);

    int blocksAvgB2 = (OUTPUT_SIZE + 16 - 1) / 16;
    avg_rows_kernel<<<blocksAvgB2, 16>>>(gpu->d_dZ2, gpu->d_db2, OUTPUT_SIZE, samples);

    dim3 dZ1blocks((samples + 15) / 16, (N_NEURONS + 15) / 16);
    naive_matmul_at_b_kernel<<<dZ1blocks, dim3(16, 16)>>>(gpu->d_W2, gpu->d_dZ2, gpu->d_dZ1, OUTPUT_SIZE, N_NEURONS, samples);

    int reluDerBlocks = (N_NEURONS * samples + 255) / 256;
    relu_derivative_kernel<<<reluDerBlocks, 256>>>(gpu->d_Z1, gpu->d_dReLU, N_NEURONS * samples);

    int dZ1Blocks = (N_NEURONS * samples + 63) / 64;
    computedZ1<<<dZ1Blocks, 64>>>(gpu->d_dZ1, gpu->d_dReLU, N_NEURONS * samples);

    dim3 dW1blocks((INPUT_SIZE + 15) / 16, (N_NEURONS + 15) / 16);
    naive_matmul_a_bt_kernel<<<dW1blocks, dim3(16, 16)>>>(gpu->d_dZ1, gpu->d_X, gpu->d_dW1, N_NEURONS, samples, INPUT_SIZE);

    int scaledW1blocks = (INPUT_SIZE * N_NEURONS + 64 - 1) / 64;
    scale_kernel<<<scaledW1blocks, 64>>>(gpu->d_dW1, 1.0 / samples, INPUT_SIZE * N_NEURONS);

    int blocksAvgB1 = (N_NEURONS + 256 - 1) / 256;
    avg_rows_kernel<<<blocksAvgB1, 256>>>(gpu->d_dZ1, gpu->d_db1, N_NEURONS, samples);
}

void update_params_gpu(GPUMemory *gpu)
{
    int blocks_W1 = (N_NEURONS * INPUT_SIZE + 255) / 256;
    update_params_kernel<<<blocks_W1, 256>>>(gpu->d_W1, gpu->d_dW1, LEARNING_RATE, N_NEURONS * INPUT_SIZE);

    int blocks_b1 = (N_NEURONS + 255) / 256;
    update_params_kernel<<<blocks_b1, 256>>>(gpu->d_b1, gpu->d_db1, LEARNING_RATE, N_NEURONS);

    int blocks_W2 = (OUTPUT_SIZE * N_NEURONS + 255) / 256;
    update_params_kernel<<<blocks_W2, 256>>>(gpu->d_W2, gpu->d_dW2, LEARNING_RATE, OUTPUT_SIZE * N_NEURONS);

    int blocks_b2 = (OUTPUT_SIZE + 255) / 256;
    update_params_kernel<<<blocks_b2, 256>>>(gpu->d_b2, gpu->d_db2, LEARNING_RATE, OUTPUT_SIZE);
}

void gradient_descent(float *X, int *Y, float *W1, float *W2, float *b1, float *b2, int samples)
{
    double start_time, end_time;
    double forward_times = 0, backward_times = 0;

    GPUMemory gpu;
    allocate_gpu_memory(&gpu, samples);

    CUDA_CHECK(cudaMemcpy(gpu.d_W1, W1, N_NEURONS * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_W2, W2, OUTPUT_SIZE * N_NEURONS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_b1, b1, N_NEURONS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_b2, b2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_X, X, samples * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_Y, Y, samples * sizeof(int), cudaMemcpyHostToDevice));

    float *A2 = (float *)malloc(OUTPUT_SIZE * samples * sizeof(float));
    int *predictions = (int *)malloc(samples * sizeof(int));

    start_time = get_time_sec();
    for (int i = 0; i < ITERATIONS; i++)
    {
        double start_fwd = get_time_sec();
        forward_prop_gpu(&gpu, samples);
        CUDA_CHECK(cudaDeviceSynchronize());
        forward_times += get_time_sec() - start_fwd;

        double start_bwd = get_time_sec();
        backward_prop_gpu(&gpu, samples);
        CUDA_CHECK(cudaDeviceSynchronize());
        backward_times += get_time_sec() - start_bwd;

        update_params_gpu(&gpu);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaMemcpy(W1, gpu.d_W1, N_NEURONS * INPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(W2, gpu.d_W2, OUTPUT_SIZE * N_NEURONS * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b1, gpu.d_b1, N_NEURONS * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b2, gpu.d_b2, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    end_time = get_time_sec();
    printf("Average forward propagation time: %f\n", (forward_times / ITERATIONS));
    printf("Average backward propagation time: %f\n", (backward_times / ITERATIONS));
    printf("Total training time: %f\n", end_time - start_time);

    free_gpu_memory(&gpu);
    free(A2);
    free(predictions);
}

void forward_prop_test(float *W1, float *W2, float *b1, float *b2, float *X, float *A2, int samples)
{
    GPUMemory gpu;
    allocate_gpu_memory(&gpu, samples);

    CUDA_CHECK(cudaMemcpy(gpu.d_W1, W1, N_NEURONS * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_W2, W2, OUTPUT_SIZE * N_NEURONS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_b1, b1, N_NEURONS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_b2, b2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_X, X, samples * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    forward_prop_gpu(&gpu, samples);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(A2, gpu.d_A2, OUTPUT_SIZE * samples * sizeof(float), cudaMemcpyDeviceToHost));

    free_gpu_memory(&gpu);
}

int main()
{
    float *X_train = (float *)malloc(INPUT_SIZE * N_TRAIN_SAMPLES * sizeof(float));
    int *Y_train = (int *)malloc(N_TRAIN_SAMPLES * sizeof(int));
    float *X_test = (float *)malloc(INPUT_SIZE * N_TEST_SAMPLES * sizeof(float));
    int *Y_test = (int *)malloc(N_TEST_SAMPLES * sizeof(int));

    loadData(X_train, Y_train, X_test, Y_test);

    float *W1 = (float *)malloc(N_NEURONS * INPUT_SIZE * sizeof(float));
    float *W2 = (float *)malloc(OUTPUT_SIZE * N_NEURONS * sizeof(float));
    float *b1 = (float *)malloc(N_NEURONS * sizeof(float));
    float *b2 = (float *)malloc(OUTPUT_SIZE * sizeof(float));

    init_data(W1, W2, b1, b2);
    gradient_descent(X_train, Y_train, W1, W2, b1, b2, N_TRAIN_SAMPLES);

    float *A2_test = (float *)malloc(OUTPUT_SIZE * N_TEST_SAMPLES * sizeof(float));
    forward_prop_test(W1, W2, b1, b2, X_test, A2_test, N_TEST_SAMPLES);

    int *predictions_test = (int *)malloc(N_TEST_SAMPLES * sizeof(int));

    get_predictions(A2_test, predictions_test, N_TEST_SAMPLES);
    float acc = get_accuracy(predictions_test, Y_test, N_TEST_SAMPLES);
    printf("Test accuracy: %f\n", acc);

    free(X_train);
    free(Y_train);
    free(X_test);
    free(Y_test);
    free(W1);
    free(W2);
    free(b1);
    free(b2);
    free(A2_test);
    free(predictions_test);
    return 0;
}