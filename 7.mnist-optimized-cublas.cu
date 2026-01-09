#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
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
#define CUBLAS_CHECK(call)                                                              \
    do                                                                                  \
    {                                                                                   \
        cublasStatus_t status = call;                                                   \
        if (status != CUBLAS_STATUS_SUCCESS)                                            \
        {                                                                               \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE);                                                         \
        }                                                                               \
    } while (0)

typedef struct
{
    float *d_W1, *d_W2, *d_b1, *d_b2;
    float *d_X, *d_Z1, *d_A1, *d_Z2, *d_A2;
    float *d_dW1, *d_dW2, *d_db1, *d_db2;
    float *d_dZ1, *d_dZ2, *d_dReLU;
    float *d_Y_one_hot;
    int *d_Y;

    cublasHandle_t cublas_handle;
} GPUMemory;

__global__ void relu_kernel(float *Z, float *A, int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size)
        A[idx] = fmaxf(Z[idx], 0.0);
}

__global__ void softmax_kernel(float *Z, float *A, int samples)
{
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_idx >= samples)
        return;

    int offset = sample_idx * OUTPUT_SIZE;
    float sum = 0.0f;

    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        float exp_val = expf(Z[offset + i]);
        A[offset + i] = exp_val;
        sum += exp_val;
    }

    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        A[offset + i] /= sum;
    }
}

__global__ void add_bias_kernel(float *x, float *bias, int batch, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * size)
    {
        int bias_idx = idx % size;
        x[idx] += bias[bias_idx];
    }
}

__global__ void one_hot_kernel(int *labels, float *one_hot, int samples)
{
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_idx >= samples)
        return;

    int label = labels[sample_idx];
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        one_hot[sample_idx * OUTPUT_SIZE + i] = (i == label) ? 1.0f : 0.0f;
    }
}

__global__ void compute_dZ2_kernel(float *probs, float *one_hot, float *grad, int samples)
{
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_idx >= samples)
        return;

    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        int idx = sample_idx * OUTPUT_SIZE + i;
        grad[idx] = probs[idx] - one_hot[idx];
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

__global__ void bias_backward_kernel(float *grad_output, float *grad_bias, int batch, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * size)
    {
        int bias_idx = idx % size;
        atomicAdd(&grad_bias[bias_idx], grad_output[idx]);
    }
}

void forward_prop_gpu(GPUMemory *gpu, int samples)
{
    const float alpha = 1.0f, beta = 0.0f;

    CUBLAS_CHECK(cublasSgemm(gpu->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             N_NEURONS, samples, INPUT_SIZE,
                             &alpha, gpu->d_W1, N_NEURONS,
                             gpu->d_X, INPUT_SIZE, &beta,
                             gpu->d_Z1, N_NEURONS));

    int total_hidden = samples * N_NEURONS;
    int grid_hidden = (total_hidden + 255) / 256;
    add_bias_kernel<<<grid_hidden, 256>>>(gpu->d_Z1, gpu->d_b1, samples, N_NEURONS);

    relu_kernel<<<grid_hidden, 256>>>(gpu->d_Z1, gpu->d_A1, N_NEURONS * samples);

    CUBLAS_CHECK(cublasSgemm(gpu->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             OUTPUT_SIZE, samples, N_NEURONS,
                             &alpha, gpu->d_W2, OUTPUT_SIZE,
                             gpu->d_A1, N_NEURONS, &beta,
                             gpu->d_Z2, OUTPUT_SIZE));

    int total_out = samples * OUTPUT_SIZE;
    int grid_out = (total_out + 255) / 256;
    add_bias_kernel<<<grid_out, 256>>>(gpu->d_Z2, gpu->d_b2, samples, OUTPUT_SIZE);

    int grid_softmax = (samples + 255) / 256;
    softmax_kernel<<<grid_softmax, 256>>>(gpu->d_Z2, gpu->d_A2, samples);
}

__global__ void avg_rows_kernel(float *input, float *output, int rows, int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows)
    {
        float sum = 0.0;
        for (int col = 0; col < cols; col++)
        {
            sum += input[col * rows + row];
        }
        output[row] = sum / cols;
    }
}

void backward_prop_gpu(GPUMemory *gpu, int samples)
{
    const float alpha = 1.0f, beta = 0.0f, scaled = 1.0f / samples;

    CUDA_CHECK(cudaMemset(gpu->d_Y_one_hot, 0, OUTPUT_SIZE * samples * sizeof(float)));

    int threads = 256;
    int blocks = (samples + threads - 1) / threads;
    one_hot_kernel<<<blocks, threads>>>(gpu->d_Y, gpu->d_Y_one_hot, samples);

    compute_dZ2_kernel<<<blocks, threads>>>(gpu->d_A2, gpu->d_Y_one_hot, gpu->d_dZ2, samples);

    CUBLAS_CHECK(cublasSgemm(gpu->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             OUTPUT_SIZE, N_NEURONS, samples,
                             &scaled, gpu->d_dZ2, OUTPUT_SIZE,
                             gpu->d_A1, N_NEURONS, &beta,
                             gpu->d_dW2, OUTPUT_SIZE));

    int blocksAvgB2 = (OUTPUT_SIZE + 16 - 1) / 16;
    avg_rows_kernel<<<blocksAvgB2, 16>>>(gpu->d_dZ2, gpu->d_db2, OUTPUT_SIZE, samples);

    CUBLAS_CHECK(cublasSgemm(gpu->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             N_NEURONS, samples, OUTPUT_SIZE,
                             &alpha, gpu->d_W2, OUTPUT_SIZE,
                             gpu->d_dZ2, OUTPUT_SIZE, &beta,
                             gpu->d_dZ1, N_NEURONS));

    int reluDerBlocks = (N_NEURONS * samples + 255) / 256;
    relu_derivative_kernel<<<reluDerBlocks, 256>>>(gpu->d_Z1, gpu->d_dReLU, N_NEURONS * samples);

    int dZ1Blocks = (N_NEURONS * samples + 63) / 64;
    computedZ1<<<dZ1Blocks, 64>>>(gpu->d_dZ1, gpu->d_dReLU, N_NEURONS * samples);

    CUBLAS_CHECK(cublasSgemm(gpu->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             N_NEURONS, INPUT_SIZE, samples,
                             &scaled, gpu->d_dZ1, N_NEURONS,
                             gpu->d_X, INPUT_SIZE, &beta,
                             gpu->d_dW1, N_NEURONS));

    int blocksAvgB1 = (N_NEURONS + 256 - 1) / 256;
    avg_rows_kernel<<<blocksAvgB1, 256>>>(gpu->d_dZ1, gpu->d_db1, N_NEURONS, samples);
}

void update_params_gpu(GPUMemory *gpu)
{
    float neg_lr = -LEARNING_RATE;

    CUBLAS_CHECK(cublasSaxpy(gpu->cublas_handle, INPUT_SIZE * N_NEURONS,
                             &neg_lr, gpu->d_dW1, 1, gpu->d_W1, 1));
    CUBLAS_CHECK(cublasSaxpy(gpu->cublas_handle, N_NEURONS * OUTPUT_SIZE,
                             &neg_lr, gpu->d_dW2, 1, gpu->d_W2, 1));
    CUBLAS_CHECK(cublasSaxpy(gpu->cublas_handle, N_NEURONS,
                             &neg_lr, gpu->d_db1, 1, gpu->d_b1, 1));
    CUBLAS_CHECK(cublasSaxpy(gpu->cublas_handle, OUTPUT_SIZE,
                             &neg_lr, gpu->d_db2, 1, gpu->d_b2, 1));

    CUDA_CHECK(cudaDeviceSynchronize());
}

void loadData_rowMajor(float *X_train, int *Y_train, float *X_test, int *Y_test)
{
    int total_samples = N_TRAIN_SAMPLES + N_TEST_SAMPLES;
    float *X_all = (float *)malloc(INPUT_SIZE * total_samples * sizeof(float));
    int *Y_all = (int *)malloc(total_samples * sizeof(int));

    FILE *file = fopen("./data/train.csv", "r");
    if (!file)
    {
        perror("train.csv");
        exit(1);
    }

    char header[100000];
    if (fgets(header, sizeof(header), file) == NULL)
    {
        fprintf(stderr, "Error reading header\n");
        exit(1);
    }

    for (int row = 0; row < total_samples; row++)
    {
        int label;
        if (fscanf(file, "%d,", &label) != 1)
        {
            fprintf(stderr, "Error reading label at row %d\n", row);
            exit(1);
        }
        Y_all[row] = label;

        for (int col = 0; col < INPUT_SIZE; col++)
        {
            float feature;
            if (fscanf(file, "%f,", &feature) != 1)
            {
                fprintf(stderr, "Error reading feature %d at row %d\n", col, row);
                exit(1);
            }
            X_all[row * INPUT_SIZE + col] = feature / 255.0f;
        }
    }
    fclose(file);

    srand(time(NULL));
    for (int i = total_samples - 1; i > 0; i--)
    {
        int j = rand() % (i + 1);

        int temp_label = Y_all[i];
        Y_all[i] = Y_all[j];
        Y_all[j] = temp_label;

        for (int k = 0; k < INPUT_SIZE; k++)
        {
            float temp = X_all[i * INPUT_SIZE + k];
            X_all[i * INPUT_SIZE + k] = X_all[j * INPUT_SIZE + k];
            X_all[j * INPUT_SIZE + k] = temp;
        }
    }

    memcpy(X_test, X_all, N_TEST_SAMPLES * INPUT_SIZE * sizeof(float));
    memcpy(Y_test, Y_all, N_TEST_SAMPLES * sizeof(int));
    memcpy(X_train, X_all + N_TEST_SAMPLES * INPUT_SIZE, N_TRAIN_SAMPLES * INPUT_SIZE * sizeof(float));
    memcpy(Y_train, Y_all + N_TEST_SAMPLES, N_TRAIN_SAMPLES * sizeof(int));

    free(X_all);
    free(Y_all);
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
    CUBLAS_CHECK(cublasCreate(&gpu->cublas_handle));
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
    CUBLAS_CHECK(cublasDestroy(gpu->cublas_handle));
}

void get_predictions_col_major(float *A2, int *predictions, int samples)
{
    for (int s = 0; s < samples; s++)
    {
        float max_val = A2[s * OUTPUT_SIZE];
        int max_idx = 0;

        for (int c = 1; c < OUTPUT_SIZE; c++)
        {
            float v = A2[s * OUTPUT_SIZE + c];
            if (v > max_val)
            {
                max_val = v;
                max_idx = c;
            }
        }
        predictions[s] = max_idx;
    }
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

    loadData_rowMajor(X_train, Y_train, X_test, Y_test);

    float *W1 = (float *)malloc(N_NEURONS * INPUT_SIZE * sizeof(float));
    float *W2 = (float *)malloc(OUTPUT_SIZE * N_NEURONS * sizeof(float));
    float *b1 = (float *)malloc(N_NEURONS * sizeof(float));
    float *b2 = (float *)malloc(OUTPUT_SIZE * sizeof(float));

    init_data(W1, W2, b1, b2);
    gradient_descent(X_train, Y_train, W1, W2, b1, b2, N_TRAIN_SAMPLES);

    float *A2_test = (float *)malloc(OUTPUT_SIZE * N_TEST_SAMPLES * sizeof(float));
    forward_prop_test(W1, W2, b1, b2, X_test, A2_test, N_TEST_SAMPLES);

    int *predictions_test = (int *)malloc(N_TEST_SAMPLES * sizeof(int));

    get_predictions_col_major(A2_test, predictions_test, N_TEST_SAMPLES);
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

    return 0;
}