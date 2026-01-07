#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.c"
#define INPUT_SIZE 784
#define OUTPUT_SIZE 10
#define N_NEURONS 256
#define ITERATIONS 10
#define LEARNING_RATE 0.1
#define N_TRAIN_SAMPLES 41000
#define N_TEST_SAMPLES 1000

void ReLu(float *Z, int size, float *A)
{
    for (int i = 0; i < size; i++)
    {
        A[i] = Z[i] > 0 ? Z[i] : 0;
    }
}

void ReLu_der(float *Z, int size, float *A)
{
    for (int i = 0; i < size; i++)
    {
        A[i] = Z[i] > 0;
    }
}

void softmax(float *Z, float *A, int samples)
{
    for (int j = 0; j < samples; j++)
    {
        float sum = 0.0f;
        for (int i = 0; i < OUTPUT_SIZE; i++)
        {
            A[i * samples + j] = expf(Z[i * samples + j]);
            sum += A[i * samples + j];
        }
        for (int i = 0; i < OUTPUT_SIZE; i++)
        {
            A[i * samples + j] /= sum;
        }
    }
}

void naive_matmul(float *A, float *B, float *C, int m, int n, int k)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
        {
            float sum = 0.0f;
            for (int l = 0; l < n; l++)
            {
                sum += A[i * n + l] * B[l * k + j];
            }
            C[i * k + j] = sum;
        }
    }
}

void naive_matmul_a_bt(float *A, float *B, float *C, int m, int n, int k)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
        {
            float sum = 0.0f;
            for (int l = 0; l < n; l++)
            {
                sum += A[i * n + l] * B[j * n + l];
            }
            C[i * k + j] = sum;
        }
    }
}

void naive_matmul_at_b(float *A, float *B, float *C, int m, int n, int k)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < k; j++)
        {
            float sum = 0.0f;
            for (int l = 0; l < m; l++)
            {
                sum += A[l * n + i] * B[l * k + j];
            }
            C[i * k + j] = sum;
        }
    }
}

void add_bias(float *Z, float *b, int neurons, int samples)
{
    for (int i = 0; i < neurons; i++)
    {
        for (int j = 0; j < samples; j++)
        {
            Z[i * samples + j] += b[i];
        }
    }
}

void forward_prop(float *W1, float *W2, float *b1, float *b2, float *X, float *A1, float *A2, float *Z1, float *Z2, int samples)
{
    naive_matmul(W1, X, Z1, N_NEURONS, INPUT_SIZE, samples);
    add_bias(Z1, b1, N_NEURONS, samples);
    ReLu(Z1, N_NEURONS * samples, A1);
    naive_matmul(W2, A1, Z2, OUTPUT_SIZE, N_NEURONS, samples);
    add_bias(Z2, b2, OUTPUT_SIZE, samples);
    softmax(Z2, A2, samples);
}

void one_hot(int *Y, int samples, float *Y_one_hot)
{
    for (int i = 0; i < OUTPUT_SIZE * samples; i++)
    {
        Y_one_hot[i] = 0.0f;
    }

    for (int sample = 0; sample < samples; sample++)
    {
        int label = Y[sample];
        Y_one_hot[label * samples + sample] = 1.0f;
    }
}

void backward_prop(float *Z1, float *A1, float *Z2, float *A2, float *W1, float *W2, float *X, int *Y, int samples, float *dZ2, float *dZ1, float *dW2, float *dW1, float *db2, float *db1, float *one_hot_Y, float *dReLU)
{
    one_hot(Y, samples, one_hot_Y);
    for (int i = 0; i < samples * OUTPUT_SIZE; i++)
    {
        dZ2[i] = A2[i] - one_hot_Y[i];
    }
    naive_matmul_a_bt(dZ2, A1, dW2, OUTPUT_SIZE, samples, N_NEURONS);
    for (int i = 0; i < OUTPUT_SIZE * N_NEURONS; i++)
    {
        dW2[i] /= samples;
    }
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        float sum = 0.0f;
        for (int j = 0; j < samples; j++)
        {
            sum += dZ2[i * samples + j];
        }
        db2[i] = sum / samples;
    }
    naive_matmul_at_b(W2, dZ2, dZ1, OUTPUT_SIZE, N_NEURONS, samples);

    ReLu_der(Z1, N_NEURONS * samples, dReLU);

    for (int i = 0; i < N_NEURONS * samples; i++)
        dZ1[i] *= dReLU[i];
    naive_matmul_a_bt(dZ1, X, dW1, N_NEURONS, samples, INPUT_SIZE);
    for (int i = 0; i < INPUT_SIZE * N_NEURONS; i++)
    {
        dW1[i] /= samples;
    }
    for (int i = 0; i < N_NEURONS; i++)
    {
        float sum = 0.0f;
        for (int j = 0; j < samples; j++)
        {
            sum += dZ1[i * samples + j];
        }
        db1[i] = sum / samples;
    }
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

void gradient_descent(float *X, int *Y, float *W1, float *W2, float *b1, float *b2, int samples)
{
    double start_time, end_time;
    double forward_times = 0, backward_times = 0;

    float *A1 = malloc(N_NEURONS * samples * sizeof(float));
    float *Z1 = malloc(N_NEURONS * samples * sizeof(float));
    float *A2 = malloc(OUTPUT_SIZE * samples * sizeof(float));
    float *Z2 = malloc(OUTPUT_SIZE * samples * sizeof(float));
    float *dZ2 = malloc(OUTPUT_SIZE * samples * sizeof(float));
    float *dW2 = malloc(OUTPUT_SIZE * N_NEURONS * sizeof(float));
    float *db2 = malloc(OUTPUT_SIZE * sizeof(float));
    float *dZ1 = malloc(N_NEURONS * samples * sizeof(float));
    float *dW1 = malloc(INPUT_SIZE * N_NEURONS * sizeof(float));
    float *db1 = malloc(N_NEURONS * sizeof(float));
    float *one_hot_Y = malloc(OUTPUT_SIZE * samples * sizeof(float));
    float *dReLU = malloc(N_NEURONS * samples * sizeof(float));

    int *predictions = malloc(samples * sizeof(int));

    start_time = get_time_sec();
    for (int i = 0; i < ITERATIONS; i++)
    {
        double start_fwd = get_time_sec();
        forward_prop(W1, W2, b1, b2, X, A1, A2, Z1, Z2, samples);
        forward_times += get_time_sec() - start_fwd;

        double start_bwd = get_time_sec();
        backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y, samples, dZ2, dZ1, dW2, dW1, db2, db1, one_hot_Y, dReLU);
        backward_times += get_time_sec() - start_bwd;

        update_params(W1, b1, W2, b2, dW1, db1, dW2, db2);
    }

    end_time = get_time_sec();

    printf("Average forward propagation time: %fs\n", (forward_times / ITERATIONS));
    printf("Average backward propagation time: %fs\n", (backward_times / ITERATIONS));
    printf("Total training time: %fs\n", (end_time - start_time));

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
    free(one_hot_Y);
    free(dReLU);
    free(predictions);
}

int main()
{
    float *X_train = malloc(INPUT_SIZE * N_TRAIN_SAMPLES * sizeof(float));
    int *Y_train = malloc(N_TRAIN_SAMPLES * sizeof(int));
    float *X_test = malloc(INPUT_SIZE * N_TEST_SAMPLES * sizeof(float));
    int *Y_test = malloc(N_TEST_SAMPLES * sizeof(int));

    loadData(X_train, Y_train, X_test, Y_test);

    float *W1 = malloc(N_NEURONS * INPUT_SIZE * sizeof(float));
    float *W2 = malloc(OUTPUT_SIZE * N_NEURONS * sizeof(float));
    float *b1 = malloc(N_NEURONS * sizeof(float));
    float *b2 = malloc(OUTPUT_SIZE * sizeof(float));

    init_data(W1, W2, b1, b2);
    gradient_descent(X_train, Y_train, W1, W2, b1, b2, N_TRAIN_SAMPLES);

    float *A1_test = malloc(N_NEURONS * N_TEST_SAMPLES * sizeof(float));
    float *Z1_test = malloc(N_NEURONS * N_TEST_SAMPLES * sizeof(float));
    float *A2_test = malloc(OUTPUT_SIZE * N_TEST_SAMPLES * sizeof(float));
    float *Z2_test = malloc(OUTPUT_SIZE * N_TEST_SAMPLES * sizeof(float));
    forward_prop(W1, W2, b1, b2, X_test, A1_test, A2_test, Z1_test, Z2_test, N_TEST_SAMPLES);

    int *predictions_test = malloc(N_TEST_SAMPLES * sizeof(int));

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