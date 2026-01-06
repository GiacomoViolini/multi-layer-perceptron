#define _GNU_SOURCE
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define INPUT_SIZE 784
#define OUTPUT_SIZE 10
#define N_NEURONS 256
#define N_TRAIN_SAMPLES 41000
#define N_TEST_SAMPLES 1000

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
    srand(42);
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

double get_time_sec()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}