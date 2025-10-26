#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
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
    float *d_dW1, *d_dW2, *d_db1, *d_db2;
    float *d_Z1, *d_Z2, *d_dZ1, *d_dZ2;

    float *d_X;
    float *h_Z2;
    float *h_dZ2;

    cublasHandle_t cublas_handle;
} GPUMemory;

__global__ void add_bias_kernel(float *x, float *bias, int batch, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * size)
    {
        int bias_idx = idx % size;
        x[idx] += bias[bias_idx];
    }
}

__global__ void relu_kernel(float *x, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total)
    {
        x[idx] = fmaxf(0.0f, x[idx]);
    }
}

__global__ void relu_backward_kernel(float *grad, float *x, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total)
    {
        grad[idx] *= (x[idx] > 0.0f ? 1.0f : 0.0f);
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

    relu_kernel<<<grid_hidden, 256>>>(gpu->d_Z1, total_hidden);

    CUBLAS_CHECK(cublasSgemm(gpu->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             OUTPUT_SIZE, samples, N_NEURONS,
                             &alpha, gpu->d_W2, OUTPUT_SIZE,
                             gpu->d_Z1, N_NEURONS, &beta,
                             gpu->d_Z2, OUTPUT_SIZE));

    int total_out = samples * OUTPUT_SIZE;
    int grid_out = (total_out + 255) / 256;
    add_bias_kernel<<<grid_out, 256>>>(gpu->d_Z2, gpu->d_b2, samples, OUTPUT_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void backward_prop_gpu(GPUMemory *gpu, int samples)
{
    const float alpha = 1.0f, beta = 0.0f;

    CUDA_CHECK(cudaMemset(gpu->d_dW1, 0, INPUT_SIZE * N_NEURONS * sizeof(float)));
    CUDA_CHECK(cudaMemset(gpu->d_dW2, 0, N_NEURONS * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemset(gpu->d_db1, 0, N_NEURONS * sizeof(float)));
    CUDA_CHECK(cudaMemset(gpu->d_db2, 0, OUTPUT_SIZE * sizeof(float)));

    CUBLAS_CHECK(cublasSgemm(gpu->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             OUTPUT_SIZE, N_NEURONS, samples,
                             &alpha, gpu->d_dZ2, OUTPUT_SIZE,
                             gpu->d_Z1, N_NEURONS, &beta,
                             gpu->d_dW2, OUTPUT_SIZE));

    int total_out = samples * OUTPUT_SIZE;
    int grid_out = (total_out + 255) / 256;
    bias_backward_kernel<<<grid_out, 256>>>(gpu->d_dZ2, gpu->d_db2, samples, OUTPUT_SIZE);

    CUBLAS_CHECK(cublasSgemm(gpu->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             N_NEURONS, samples, OUTPUT_SIZE,
                             &alpha, gpu->d_W2, OUTPUT_SIZE,
                             gpu->d_dZ2, OUTPUT_SIZE, &beta,
                             gpu->d_dZ1, N_NEURONS));

    int total_hidden = samples * N_NEURONS;
    int grid_hidden = (total_hidden + 255) / 256;
    relu_backward_kernel<<<grid_hidden, 256>>>(gpu->d_dZ1, gpu->d_Z1, total_hidden);

    CUBLAS_CHECK(cublasSgemm(gpu->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             N_NEURONS, INPUT_SIZE, samples,
                             &alpha, gpu->d_dZ1, N_NEURONS,
                             gpu->d_X, INPUT_SIZE, &beta,
                             gpu->d_dW1, N_NEURONS));

    bias_backward_kernel<<<grid_hidden, 256>>>(gpu->d_dZ1, gpu->d_db1, samples, N_NEURONS);
}

void update_weights_only(GPUMemory *gpu, float lr)
{
    float neg_lr = -lr;

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

void softmax_and_gradZ2(int samples, float *h_logits, int *labels, float *h_grad)
{
    for (int b = 0; b < samples; b++)
    {
        float *logits = h_logits + b * OUTPUT_SIZE;
        int label = labels[b];
        float max_logit = -INFINITY;
        for (int i = 0; i < OUTPUT_SIZE; i++)
        {
            if (logits[i] > max_logit)
                max_logit = logits[i];
        }
        float sum_exp = 0.0f;
        for (int i = 0; i < OUTPUT_SIZE; i++)
        {
            float shifted = logits[i] - max_logit;
            float expv = expf(shifted);
            sum_exp += expv;
            h_grad[b * OUTPUT_SIZE + i] = expv;
        }
        for (int i = 0; i < OUTPUT_SIZE; i++)
        {
            h_grad[b * OUTPUT_SIZE + i] /= sum_exp;
        }
        h_grad[b * OUTPUT_SIZE + label] -= 1.0f;
    }
    for (int i = 0; i < samples * OUTPUT_SIZE; i++)
    {
        h_grad[i] /= samples;
    }
}

float compute_accuracy(GPUMemory *gpu, float *data, int *labels, int samples)
{
    int correct = 0;
    CUDA_CHECK(cudaMemcpy(gpu->d_X, data,
                          samples * INPUT_SIZE * sizeof(float),
                          cudaMemcpyHostToDevice));

    forward_prop_gpu(gpu, samples);
    CUDA_CHECK(cudaMemcpy(gpu->h_Z2, gpu->d_Z2,
                          samples * OUTPUT_SIZE * sizeof(float),
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < samples; i++)
    {
        float *logits = gpu->h_Z2 + i * OUTPUT_SIZE;
        int pred = 0;
        float max_val = logits[0];

        for (int j = 1; j < OUTPUT_SIZE; j++)
        {
            if (logits[j] > max_val)
            {
                max_val = logits[j];
                pred = j;
            }
        }

        if (pred == labels[i])
        {
            correct++;
        }
    }

    return ((float)correct / (float)samples);
}

void loadData(float *X_train, int *Y_train, float *X_test, int *Y_test)
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

    // Read data in row-major format: X[row * INPUT_SIZE + col]
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

    // Shuffle samples (Fisher-Yates)
    srand(time(NULL));
    for (int i = total_samples - 1; i > 0; i--)
    {
        int j = rand() % (i + 1);

        // Swap labels
        int temp_label = Y_all[i];
        Y_all[i] = Y_all[j];
        Y_all[j] = temp_label;

        // Swap feature rows
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
    CUDA_CHECK(cudaMalloc(&gpu->d_Z1, N_NEURONS * samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu->d_Z2, OUTPUT_SIZE * samples * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&gpu->d_dW1, N_NEURONS * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu->d_dW2, OUTPUT_SIZE * N_NEURONS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu->d_db1, N_NEURONS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu->d_db2, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu->d_dZ1, N_NEURONS * samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu->d_dZ2, OUTPUT_SIZE * samples * sizeof(float)));

    gpu->h_Z2 = (float *)malloc(N_TRAIN_SAMPLES * OUTPUT_SIZE * sizeof(float));
    gpu->h_dZ2 = (float *)malloc(N_TRAIN_SAMPLES * OUTPUT_SIZE * sizeof(float));

    CUBLAS_CHECK(cublasCreate(&gpu->cublas_handle));
}

void free_gpu_memory(GPUMemory *gpu)
{
    CUDA_CHECK(cudaFree(gpu->d_W1));
    CUDA_CHECK(cudaFree(gpu->d_W2));
    CUDA_CHECK(cudaFree(gpu->d_b1));
    CUDA_CHECK(cudaFree(gpu->d_b2));
    CUDA_CHECK(cudaFree(gpu->d_X));
    CUDA_CHECK(cudaFree(gpu->d_Z1));
    CUDA_CHECK(cudaFree(gpu->d_Z2));
    CUDA_CHECK(cudaFree(gpu->d_dW1));
    CUDA_CHECK(cudaFree(gpu->d_dW2));
    CUDA_CHECK(cudaFree(gpu->d_db1));
    CUDA_CHECK(cudaFree(gpu->d_db2));
    CUDA_CHECK(cudaFree(gpu->d_dZ1));
    CUDA_CHECK(cudaFree(gpu->d_dZ2));

    free(gpu->h_Z2);
    free(gpu->h_dZ2);

    CUBLAS_CHECK(cublasDestroy(gpu->cublas_handle));
}

void init_data(float *W1, float *W2, float *b1, float *b2)
{
    float scale = sqrtf(2.0f / INPUT_SIZE);
    for (int i = 0; i < N_NEURONS * INPUT_SIZE; i++)
    {
        W1[i] = ((float)rand() / RAND_MAX) * 2.0f * scale - scale;
    }
    for (int i = 0; i < OUTPUT_SIZE * N_NEURONS; i++)
    {
        W2[i] = ((float)rand() / RAND_MAX) * 2.0f * scale - scale;
    }
    for (int i = 0; i < N_NEURONS; i++)
    {
        b1[i] = 0.0f;
    }
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        b2[i] = 0.0f;
    }
}

void gradient_descent(float *X, int *Y, int samples)
{
    clock_t start_time, end_time;
    start_time = clock();
    int forward_times = 0, backward_times = 0;

    GPUMemory gpu;
    allocate_gpu_memory(&gpu, samples);

    CUDA_CHECK(cudaMemcpy(gpu.d_X, X,
                          samples * INPUT_SIZE * sizeof(float),
                          cudaMemcpyHostToDevice));

    float *W1 = (float *)malloc(N_NEURONS * INPUT_SIZE * sizeof(float));
    float *W2 = (float *)malloc(OUTPUT_SIZE * N_NEURONS * sizeof(float));
    float *b1 = (float *)malloc(N_NEURONS * sizeof(float));
    float *b2 = (float *)malloc(OUTPUT_SIZE * sizeof(float));

    init_data(W1, W2, b1, b2);

    CUDA_CHECK(cudaMemcpy(gpu.d_W1, W1,
                          N_NEURONS * INPUT_SIZE * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_W2, W2,
                          OUTPUT_SIZE * N_NEURONS * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_b1, b1,
                          N_NEURONS * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_b2, b2,
                          OUTPUT_SIZE * sizeof(float),
                          cudaMemcpyHostToDevice));

    for (int i = 0; i < ITERATIONS; i++)
    {
        clock_t start_fwd = clock();
        forward_prop_gpu(&gpu, samples);
        forward_times += clock() - start_fwd;

        CUDA_CHECK(cudaMemcpy(gpu.h_Z2, gpu.d_Z2,
                              samples * OUTPUT_SIZE * sizeof(float),
                              cudaMemcpyDeviceToHost));

        softmax_and_gradZ2(samples, gpu.h_Z2,
                           Y, gpu.h_dZ2);

        CUDA_CHECK(cudaMemcpy(gpu.d_dZ2, gpu.h_dZ2,
                              samples * OUTPUT_SIZE * sizeof(float),
                              cudaMemcpyHostToDevice));
        clock_t start_bwd = clock();
        backward_prop_gpu(&gpu, samples);
        backward_times += clock() - start_bwd;
        update_weights_only(&gpu, LEARNING_RATE);
        if (i % 10 == 0)
        {
            float acc = compute_accuracy(&gpu, X, Y, samples);
            printf("Iteration %d, accuracy: %f\n", i, acc);
        }
    }

    end_time = clock();
    printf("Average forward propagation time: %f\n", ((float)forward_times / CLOCKS_PER_SEC) / ITERATIONS);
    printf("Average backward propagation time: %f\n", ((float)backward_times / CLOCKS_PER_SEC) / ITERATIONS);
    printf("Total training time: %f\n", (float)(end_time - start_time) / CLOCKS_PER_SEC);

    free(W1);
    free(W2);
    free(b1);
    free(b2);
    free_gpu_memory(&gpu);
}

int main()
{
    float *X_train = (float *)malloc(INPUT_SIZE * N_TRAIN_SAMPLES * sizeof(float));
    int *Y_train = (int *)malloc(N_TRAIN_SAMPLES * sizeof(int));
    float *X_test = (float *)malloc(INPUT_SIZE * N_TEST_SAMPLES * sizeof(float));
    int *Y_test = (int *)malloc(N_TEST_SAMPLES * sizeof(int));

    loadData(X_train, Y_train, X_test, Y_test);

    gradient_descent(X_train, Y_train, N_TRAIN_SAMPLES);

    free(X_train);
    free(Y_train);
    free(X_test);
    free(Y_test);

    return 0;
}
