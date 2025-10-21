#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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

void loadData(float *X_train, int *Y_train, float *X_test, int *Y_test)
{
    float *X_all = (float *)malloc(INPUT_SIZE * (N_TRAIN_SAMPLES + N_TEST_SAMPLES) * sizeof(float));
    int *Y_all = (int *)malloc((N_TRAIN_SAMPLES + N_TEST_SAMPLES) * sizeof(int));

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
    for (int row = 0; row < N_TRAIN_SAMPLES + N_TEST_SAMPLES; row++)
    {
        int label;
        fscanf(file, "%d,", &label);
        Y_all[row] = label;

        for (int col = 0; col < INPUT_SIZE; col++)
        {
            float feature;
            fscanf(file, "%f,", &feature);
            X_all[row * INPUT_SIZE + col] = feature / 255.0;
        }
    }
    fclose(file);

    // Shuffle data: https://www.geeksforgeeks.org/dsa/shuffle-a-given-array-using-fisher-yates-shuffle-algorithm/
    srand(time(NULL));
    for (int i = N_TRAIN_SAMPLES + N_TEST_SAMPLES - 1; i > 0; i--)
    {
        int j = rand() % (i + 1);

        int temp_label = Y_all[i];
        Y_all[i] = Y_all[j];
        Y_all[j] = temp_label;

        for (int k = 0; k < INPUT_SIZE; k++)
        {
            float temp_feature = X_all[i * INPUT_SIZE + k];
            X_all[i * INPUT_SIZE + k] = X_all[j * INPUT_SIZE + k];
            X_all[j * INPUT_SIZE + k] = temp_feature;
        }
    }

    for (int row = 0; row < N_TEST_SAMPLES; row++)
    {
        Y_test[row] = Y_all[row];
        for (int col = 0; col < INPUT_SIZE; col++)
            X_test[row * INPUT_SIZE + col] = X_all[row * INPUT_SIZE + col];
    }

    for (int row = 0; row < N_TRAIN_SAMPLES; row++)
    {
        Y_train[row] = Y_all[row + N_TEST_SAMPLES];
        for (int col = 0; col < INPUT_SIZE; col++)
            X_train[row * INPUT_SIZE + col] = X_all[(row + N_TEST_SAMPLES) * INPUT_SIZE + col];
    }
    free(X_all);
    free(Y_all);
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

void forward_prop(float *W1, float *W2, float *b1, float *b2, float *X, float *A1, float *A2, float *Z1, float *Z2, int samples)
{
    float *d_W1, *d_X, *d_Z1, *d_A1, *d_W2, *d_Z2, *d_b1, *d_b2, *d_A2;
    CUDA_CHECK(cudaMalloc((void **)&d_W1, N_NEURONS * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_X, samples * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_Z1, N_NEURONS * samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_A1, N_NEURONS * samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_W2, N_NEURONS * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_Z2, OUTPUT_SIZE * samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_b1, N_NEURONS * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_b2, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_A2, OUTPUT_SIZE * samples * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_W1, W1, N_NEURONS * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2, W2, N_NEURONS * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_X, X, samples * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1, b1, N_NEURONS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, b2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    dim3 threadsPerBlockFirstLayer(32, 32);
    // Col Major order
    dim3 numBlocksFirstLayer((samples + threadsPerBlockFirstLayer.x - 1) / threadsPerBlockFirstLayer.x,
                             (N_NEURONS + threadsPerBlockFirstLayer.y - 1) / threadsPerBlockFirstLayer.y);
    naive_matmul_a_bt_kernel<<<numBlocksFirstLayer, threadsPerBlockFirstLayer>>>(d_W1, d_X, d_Z1, N_NEURONS, INPUT_SIZE, samples);
    CUDA_CHECK(cudaDeviceSynchronize());

    add_bias_kernel<<<numBlocksFirstLayer, threadsPerBlockFirstLayer>>>(d_Z1, d_b1, N_NEURONS, samples);
    CUDA_CHECK(cudaDeviceSynchronize());

    int ReLuBlocks = (N_NEURONS * samples + 256 - 1) / 256;
    relu_kernel<<<ReLuBlocks, 256>>>(d_Z1, d_A1, N_NEURONS * samples);
    CUDA_CHECK(cudaDeviceSynchronize());

    dim3 threadsPerBlockSecondLayer(16, 16);
    // Col Major order
    dim3 numBlocksSecondLayer((samples + threadsPerBlockSecondLayer.x - 1) / threadsPerBlockSecondLayer.x,
                              (OUTPUT_SIZE + threadsPerBlockSecondLayer.y - 1) / threadsPerBlockSecondLayer.y);
    naive_matmul_kernel<<<numBlocksSecondLayer, threadsPerBlockSecondLayer>>>(d_W2, d_A1, d_Z2, OUTPUT_SIZE, N_NEURONS, samples);
    CUDA_CHECK(cudaDeviceSynchronize());

    add_bias_kernel<<<numBlocksSecondLayer, threadsPerBlockSecondLayer>>>(d_Z2, d_b2, OUTPUT_SIZE, samples);
    CUDA_CHECK(cudaDeviceSynchronize());

    int softmaxBlocks = (samples + 32 - 1) / 32;
    softmax_kernel<<<softmaxBlocks, 32>>>(d_Z2, d_A2, samples);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(Z1, d_Z1, N_NEURONS * samples * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(A1, d_A1, N_NEURONS * samples * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Z2, d_Z2, OUTPUT_SIZE * samples * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(A2, d_A2, OUTPUT_SIZE * samples * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_W1));
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Z1));
    CUDA_CHECK(cudaFree(d_W2));
    CUDA_CHECK(cudaFree(d_A1));
    CUDA_CHECK(cudaFree(d_Z2));
    CUDA_CHECK(cudaFree(d_b1));
    CUDA_CHECK(cudaFree(d_b2));
    CUDA_CHECK(cudaFree(d_A2));
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

void backward_prop(float *Z1, float *A1, float *Z2, float *A2, float *W1, float *W2, float *X, int *Y, int samples, float *dZ2, float *dZ1, float *dW2, float *dW1, float *db2, float *db1)
{
    int *d_Y;
    float *d_Y_one_hot, *d_A2, *d_dZ2, *d_A1, *d_dW2, *d_b2, *d_dZ1, *d_W2, *d_Z1, *d_dReLU, *d_X, *d_dW1, *d_b1;

    CUDA_CHECK(cudaMalloc(&d_Y, samples * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_Y_one_hot, OUTPUT_SIZE * samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_A2, OUTPUT_SIZE * samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dZ2, OUTPUT_SIZE * samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_A1, N_NEURONS * samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dW2, OUTPUT_SIZE * N_NEURONS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dZ1, N_NEURONS * samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W2, OUTPUT_SIZE * N_NEURONS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Z1, N_NEURONS * samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dReLU, N_NEURONS * samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_X, samples * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dW1, N_NEURONS * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1, N_NEURONS * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A2, A2, OUTPUT_SIZE * samples * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Y, Y, samples * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_Y_one_hot, 0, OUTPUT_SIZE * samples * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A1, A1, N_NEURONS * samples * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, db2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2, W2, OUTPUT_SIZE * N_NEURONS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Z1, Z1, N_NEURONS * samples * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_X, X, samples * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    int diffblocks = (samples + 64 - 1) / 64;
    one_hot_kernel<<<diffblocks, 64>>>(d_Y, samples, d_Y_one_hot);
    CUDA_CHECK(cudaDeviceSynchronize());

    int dZ2blocks = (samples * OUTPUT_SIZE + 64 - 1) / 64;
    compute_dZ2_kernel<<<dZ2blocks, 64>>>(d_A2, d_Y_one_hot, d_dZ2, OUTPUT_SIZE * samples);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Col Major order
    dim3 dW2blocks((N_NEURONS + 15) / 16, (OUTPUT_SIZE + 15) / 16);
    naive_matmul_a_bt_kernel<<<dW2blocks, dim3(16, 16)>>>(d_dZ2, d_A1, d_dW2, OUTPUT_SIZE, samples, N_NEURONS);
    CUDA_CHECK(cudaDeviceSynchronize());

    int scaleW2blocks = (OUTPUT_SIZE * N_NEURONS + 64 - 1) / 64;
    scale_kernel<<<scaleW2blocks, 64>>>(d_dW2, 1.0 / samples, OUTPUT_SIZE * N_NEURONS);
    CUDA_CHECK(cudaDeviceSynchronize());

    int blocksAvgB2 = (OUTPUT_SIZE + 16 - 1) / 16;
    avg_rows_kernel<<<blocksAvgB2, 16>>>(d_dZ2, d_b2, OUTPUT_SIZE, samples);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Col Major order
    dim3 dZ1blocks((samples + 15) / 16, (N_NEURONS + 15) / 16);
    naive_matmul_at_b_kernel<<<dZ1blocks, dim3(16, 16)>>>(d_W2, d_dZ2, d_dZ1, OUTPUT_SIZE, N_NEURONS, samples);
    CUDA_CHECK(cudaDeviceSynchronize());

    int reluDerBlocks = (N_NEURONS * samples + 255) / 256;
    relu_derivative_kernel<<<reluDerBlocks, 256>>>(d_Z1, d_dReLU, N_NEURONS * samples);
    CUDA_CHECK(cudaDeviceSynchronize());

    int dZ1Blocks = (N_NEURONS * samples + 63) / 64;
    computedZ1<<<dZ1Blocks, 64>>>(d_dZ1, d_dReLU, N_NEURONS * samples);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Col Major order
    dim3 dW1blocks((INPUT_SIZE + 15) / 16, (N_NEURONS + 15) / 16);
    naive_matmul_a_bt_kernel<<<dW1blocks, dim3(16, 16)>>>(d_dZ1, d_X, d_dW1, N_NEURONS, samples, INPUT_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    int scaledW1blocks = (INPUT_SIZE * N_NEURONS + 64 - 1) / 64;
    scale_kernel<<<scaledW1blocks, 64>>>(d_dW1, 1.0 / samples, INPUT_SIZE * N_NEURONS);
    CUDA_CHECK(cudaDeviceSynchronize());

    int blocksAvgB1 = (N_NEURONS + 256 - 1) / 256;
    avg_rows_kernel<<<blocksAvgB1, 256>>>(d_dZ1, d_b1, N_NEURONS, samples);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(dZ1, d_dZ1, N_NEURONS * samples * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dZ2, d_dZ2, OUTPUT_SIZE * samples * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dW2, d_dW2, OUTPUT_SIZE * N_NEURONS * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(db2, d_b2, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dW1, d_dW1, N_NEURONS * INPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(db1, d_b1, N_NEURONS * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_Y));
    CUDA_CHECK(cudaFree(d_Y_one_hot));
    CUDA_CHECK(cudaFree(d_A2));
    CUDA_CHECK(cudaFree(d_dZ2));
    CUDA_CHECK(cudaFree(d_A1));
    CUDA_CHECK(cudaFree(d_dW2));
    CUDA_CHECK(cudaFree(d_b2));
    CUDA_CHECK(cudaFree(d_dZ1));
    CUDA_CHECK(cudaFree(d_W2));
    CUDA_CHECK(cudaFree(d_Z1));
    CUDA_CHECK(cudaFree(d_dReLU));
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_dW1));
    CUDA_CHECK(cudaFree(d_b1));
}

void update_params(
    float *W1, float *b1, float *W2, float *b2,
    float *dW1, float *db1, float *dW2, float *db2)
{
    for (int i = 0; i < N_NEURONS * INPUT_SIZE; i++)
        W1[i] -= LEARNING_RATE * dW1[i];
    for (int i = 0; i < N_NEURONS; i++)
        b1[i] -= LEARNING_RATE * db1[i];
    for (int i = 0; i < OUTPUT_SIZE * N_NEURONS; i++)
        W2[i] -= LEARNING_RATE * dW2[i];
    for (int i = 0; i < OUTPUT_SIZE; i++)
        b2[i] -= LEARNING_RATE * db2[i];
}

void get_predictions(float *A2, int *predictions, int samples)
{
    for (int sample = 0; sample < samples; sample++)
    {
        float max_val = A2[0 * samples + sample];
        int max_idx = 0;

        for (int i = 1; i < OUTPUT_SIZE; i++)
        {
            if (A2[i * samples + sample] > max_val)
            {
                max_val = A2[i * samples + sample];
                max_idx = i;
            }
        }

        predictions[sample] = max_idx;
    }
}

float get_accuracy(int *predictions, int *Y, int samples)
{
    int correct = 0;

    for (int i = 0; i < samples; i++)
    {
        if (predictions[i] == Y[i])
            correct++;
    }

    return (float)correct / samples;
}

void gradient_descent(float *X, int *Y, float *W1, float *W2, float *b1, float *b2, int samples)
{
    clock_t start_time, end_time;
    start_time = clock();
    int forward_times = 0, backward_times = 0;

    float *A1 = (float *)malloc(N_NEURONS * samples * sizeof(float));
    float *Z1 = (float *)malloc(N_NEURONS * samples * sizeof(float));
    float *A2 = (float *)malloc(OUTPUT_SIZE * samples * sizeof(float));
    float *Z2 = (float *)malloc(OUTPUT_SIZE * samples * sizeof(float));
    float *dZ2 = (float *)malloc(OUTPUT_SIZE * samples * sizeof(float));
    float *dW2 = (float *)malloc(OUTPUT_SIZE * N_NEURONS * sizeof(float));
    float *db2 = (float *)malloc(OUTPUT_SIZE * sizeof(float));
    float *dZ1 = (float *)malloc(N_NEURONS * samples * sizeof(float));
    float *dW1 = (float *)malloc(INPUT_SIZE * N_NEURONS * sizeof(float));
    float *db1 = (float *)malloc(N_NEURONS * sizeof(float));

    int *predictions = (int *)malloc(samples * sizeof(int));

    for (int i = 0; i < ITERATIONS; i++)
    {
        clock_t start_fwd = clock();
        forward_prop(W1, W2, b1, b2, X, A1, A2, Z1, Z2, samples);
        forward_times += clock() - start_fwd;
        clock_t start_bwd = clock();
        backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y, samples, dZ2, dZ1, dW2, dW1, db2, db1);
        backward_times += clock() - start_bwd;
        update_params(W1, b1, W2, b2, dW1, db1, dW2, db2);
        if (i % 10 == 0)
        {
            get_predictions(A2, predictions, samples);
            float acc = get_accuracy(predictions, Y, samples);
            printf("Iteration %d, accuracy: %f\n", i, acc);
        }
    }
    end_time = clock();
    printf("Average forward propagation time: %f\n", ((float)forward_times / CLOCKS_PER_SEC) / ITERATIONS);
    printf("Average backward propagation time: %f\n", ((float)backward_times / CLOCKS_PER_SEC) / ITERATIONS);
    printf("Total training time: %f\n", (float)(end_time - start_time) / CLOCKS_PER_SEC);
    free(A1);
    free(Z1);
    free(A2);
    free(Z2);
    free(dZ2);
    free(dW2);
    free(db2);
    free(dZ1);
    free(dW1);
    free(db1);
    free(predictions);
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

    float *A1_test = (float *)malloc(N_NEURONS * N_TEST_SAMPLES * sizeof(float));
    float *Z1_test = (float *)malloc(N_NEURONS * N_TEST_SAMPLES * sizeof(float));
    float *A2_test = (float *)malloc(OUTPUT_SIZE * N_TEST_SAMPLES * sizeof(float));
    float *Z2_test = (float *)malloc(OUTPUT_SIZE * N_TEST_SAMPLES * sizeof(float));
    forward_prop(W1, W2, b1, b2, X_test, A1_test, A2_test, Z1_test, Z2_test, N_TEST_SAMPLES);

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
    free(A1_test);
    free(Z1_test);
    free(A2_test);
    free(Z2_test);
    free(predictions_test);
    return 0;
}