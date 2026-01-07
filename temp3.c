#define _GNU_SOURCE
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>
#define INPUT_SIZE 784
#define OUTPUT_SIZE 10
#define N_NEURONS 256
#define ITERATIONS 200
#define LEARNING_RATE 0.1
#define N_TRAIN_SAMPLES 41000
#define N_TEST_SAMPLES 1000
#define ALPHA 1.0f
#define BETA 0.0f

// gcc -Ofast -fopenmp -march=native -mtune=native -flto -funroll-loops -fno-plt -o temp3 temp3.c -lopenblas -lm
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

    // Read data in column-major format: X[col * total_samples + row]
    for (int row = 0; row < total_samples; row++)
    {
        int label;
        if (fscanf(file, "%d,", &label) != 1)
        {
            fprintf(stderr, "Error reading label at row %d\n", row);
            break;
        }
        Y_all[row] = label;

        for (int col = 0; col < INPUT_SIZE; col++)
        {
            float feature;
            if (fscanf(file, "%f,", &feature) != 1)
            {
                fprintf(stderr, "Error reading feature at row %d, col %d\n", row, col);
                break;
            }
            X_all[col * total_samples + row] = feature / 255.0;
        }
    }
    fclose(file);

    // Shuffle data: https://www.geeksforgeeks.org/dsa/shuffle-a-given-array-using-fisher-yates-shuffle-algorithm/
    for (int i = total_samples - 1; i > 0; i--)
    {
        int j = rand() % (i + 1);

        int temp_label = Y_all[i];
        Y_all[i] = Y_all[j];
        Y_all[j] = temp_label;

        for (int k = 0; k < INPUT_SIZE; k++)
        {
            float temp_feature = X_all[k * total_samples + i];
            X_all[k * total_samples + i] = X_all[k * total_samples + j];
            X_all[k * total_samples + j] = temp_feature;
        }
    }

    for (int row = 0; row < N_TEST_SAMPLES; row++)
    {
        Y_test[row] = Y_all[row];
        for (int col = 0; col < INPUT_SIZE; col++)
            X_test[col * N_TEST_SAMPLES + row] = X_all[col * total_samples + row];
    }

    for (int row = 0; row < N_TRAIN_SAMPLES; row++)
    {
        Y_train[row] = Y_all[row + N_TEST_SAMPLES];
        for (int col = 0; col < INPUT_SIZE; col++)
            X_train[col * N_TRAIN_SAMPLES + row] = X_all[col * total_samples + (row + N_TEST_SAMPLES)];
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

void blas_matmul(float *A, float *B, float *C,
                 int m, int n, int k, float alpha, float beta)
{
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans, CblasNoTrans,
        m, k, n,
        alpha,
        A, n,
        B, k,
        beta,
        C, k);
}

void blas_matmul_a_bt(float *A, float *B, float *C,
                      int m, int n, int k, float alpha, float beta)
{
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans, CblasTrans,
        m, k, n,
        alpha,
        A, n,
        B, n,
        beta,
        C, k);
}

void blas_matmul_at_b(float *A, float *B, float *C,
                      int m, int n, int k, float alpha, float beta)
{
    cblas_sgemm(
        CblasRowMajor,
        CblasTrans, CblasNoTrans,
        n, k, m,
        alpha,
        A, n,
        B, k,
        beta,
        C, k);
}

void add_bias_and_softmax(float *Z, float *b, int neurons, int samples, float *A)
{
#pragma omp parallel for schedule(static)
    for (int j = 0; j < samples; j++)
    {
        float sum = 0.0f;
        for (int i = 0; i < neurons; i++)
        {
            Z[i * samples + j] += b[i];
            A[i * samples + j] = expf(Z[i * samples + j]);
            sum += A[i * samples + j];
        }
        for (int i = 0; i < neurons; i++)
        {
            A[i * samples + j] /= sum;
        }
    }
}

void add_bias_and_ReLu(float *Z, float *b, int neurons, int samples, float *A)
{
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N_NEURONS; i++)
    {
        for (int j = 0; j < samples; j++)
        {
            float val = Z[i * samples + j] + b[i];
            A[i * samples + j] = val > 0 ? val : 0;
        }
    }
}

void forward_prop(float *W1, float *W2, float *b1, float *b2, float *X, float *A1, float *A2, float *Z1, float *Z2, int samples)
{
    blas_matmul(W1, X, Z1, N_NEURONS, INPUT_SIZE, samples, ALPHA, BETA);
    add_bias_and_ReLu(Z1, b1, N_NEURONS, samples, A1);
    blas_matmul(W2, A1, Z2, OUTPUT_SIZE, N_NEURONS, samples, ALPHA, BETA);
    add_bias_and_softmax(Z2, b2, OUTPUT_SIZE, samples, A2);
}

void one_hot_and_dZ2(int *Y, int samples, float *dZ2, float *A2)
{
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        for (int j = 0; j < samples; j++)
        {
            float y_val = (i == Y[j]) ? 1.0f : 0.0f;
            dZ2[i * samples + j] = A2[i * samples + j] - y_val;
        }
    }
}

void ReLu_der_and_dZ1(float *Z1, int size, float *dZ1)
{
#pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++)
    {
        if (Z1[i] <= 0)
        {
            dZ1[i] = 0.0f;
        }
    }
}

void backward_prop(float *Z1, float *A1, float *Z2, float *A2, float *W1, float *W2, float *X, int *Y, int samples, float *dZ2, float *dZ1, float *dW2, float *dW1, float *db2, float *db1)
{
    one_hot_and_dZ2(Y, samples, dZ2, A2);
    blas_matmul_a_bt(dZ2, A1, dW2, OUTPUT_SIZE, samples, N_NEURONS, ALPHA / samples, BETA);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        float sum = 0.0f;
        for (int j = 0; j < samples; j++)
        {
            sum += dZ2[i * samples + j];
        }
        db2[i] = sum / samples;
    }
    blas_matmul_at_b(W2, dZ2, dZ1, OUTPUT_SIZE, N_NEURONS, samples, ALPHA, BETA);
    ReLu_der_and_dZ1(Z1, N_NEURONS * samples, dZ1);
    blas_matmul_a_bt(dZ1, X, dW1, N_NEURONS, samples, INPUT_SIZE, ALPHA / samples, BETA);
#pragma omp parallel for schedule(static)
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
#pragma omp parallel
    {
        #pragma omp for nowait schedule(static)
        for (int i = 0; i < N_NEURONS * INPUT_SIZE; i++)
            W1[i] -= LEARNING_RATE * dW1[i];
        #pragma omp for nowait schedule(static)
        for (int i = 0; i < N_NEURONS; i++)
            b1[i] -= LEARNING_RATE * db1[i];
        #pragma omp for nowait schedule(static)
        for (int i = 0; i < OUTPUT_SIZE * N_NEURONS; i++)
            W2[i] -= LEARNING_RATE * dW2[i];
        #pragma omp for schedule(static)
        for (int i = 0; i < OUTPUT_SIZE; i++)
            b2[i] -= LEARNING_RATE * db2[i];
    }
}

double get_time_sec()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
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

    int *predictions = malloc(samples * sizeof(int));

    start_time = get_time_sec();
    for (int i = 0; i < ITERATIONS; i++)
    {
        double start_fwd = get_time_sec();
        forward_prop(W1, W2, b1, b2, X, A1, A2, Z1, Z2, samples);
        forward_times += get_time_sec() - start_fwd;

        double start_bwd = get_time_sec();
        backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y, samples, dZ2, dZ1, dW2, dW1, db2, db1);
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
    free(predictions);
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

int main()
{
    float *X_train = malloc(INPUT_SIZE * N_TRAIN_SAMPLES * sizeof(float));
    int *Y_train = malloc(N_TRAIN_SAMPLES * sizeof(int));
    float *X_test = malloc(INPUT_SIZE * N_TEST_SAMPLES * sizeof(float));
    int *Y_test = malloc(N_TEST_SAMPLES * sizeof(int));

    loadData(X_train, Y_train, X_test, Y_test);

    srand(42);
    for (int run = 0; run < 10; run++) {
        printf("\n--- Run %d ---\n", run + 1); 

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
        int *predictions_test = malloc(N_TEST_SAMPLES * sizeof(int));

        forward_prop(W1, W2, b1, b2, X_test, A1_test, A2_test, Z1_test, Z2_test, N_TEST_SAMPLES);
        get_predictions(A2_test, predictions_test, N_TEST_SAMPLES);
        float acc = get_accuracy(predictions_test, Y_test, N_TEST_SAMPLES);
        printf("Run %d Test accuracy: %f\n", run + 1, acc);

        free(W1); free(W2); free(b1); free(b2);
        free(A1_test); free(Z1_test); free(A2_test); free(Z2_test);
        free(predictions_test);
    }

    free(X_train); free(Y_train); free(X_test); free(Y_test);
    return 0;
}