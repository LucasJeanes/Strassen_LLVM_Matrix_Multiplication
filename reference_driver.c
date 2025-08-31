#include "reference_driver.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

// Aligned for AVX (4 * doubles or 32-byte) instructions
double* allocate_matrix(int n) {
    double* matrix = (double*)aligned_alloc(32, n * n * sizeof(double));
    if (matrix) {
        memset(matrix, 0, n * n * sizeof(double));
    }
    return matrix;
}

void free_matrix(double* matrix) {
    if (matrix) {
        free(matrix);
    }
}

void print_matrix(const char* name, double* matrix, int n) {
    printf("\n%s:\n", name);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%8.2f ", matrix[i * n + j]);
        }
        printf("\n");
    }
}

void initialize_matrix_random(double* matrix, int n, double min_val, double max_val) {
    double range = max_val - min_val;
    for (int i = 0; i < n * n; i++) {
        matrix[i] = min_val + (double)rand() / RAND_MAX * range;
    }
}

void initialize_matrix_identity(double* matrix, int n) {
    memset(matrix, 0, n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        matrix[i * n + i] = 1.0;
    }
}

void initialize_matrix_sequential(double* matrix, int n) {
    for (int i = 0; i < n * n; i++) {
        matrix[i] = (double)(i + 1);
    }
}

int verify_matrices_equal(double* A, double* B, int n, double tolerance) {
    double max_diff = 0.0;
    int errors = 0;
    
    for (int i = 0; i < n * n; i++) {
        double diff = fabs(A[i] - B[i]);
        if (diff > tolerance) {
            errors++;
            if (errors <= 5) { // Print first 5 errors
                int row = i / n;
                int col = i % n;
                printf("  Error at (%d,%d): %.10f vs %.10f (diff: %.2e)\n", 
                       row, col, A[i], B[i], diff);
            }
        }
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    
    printf("  Max difference: %.2e, Errors: %d/%d\n", max_diff, errors, n * n);
    return errors == 0;
}

void reference_multiply(double* A, double* B, double* C, int n) {
    // Standard O(n^3) matrix multiplication
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

double benchmark_function(void (*func)(double*, double*, double*, int), 
                         double* A, double* B, double* C, int n, int iterations) {
    clock_t start, end;
    double total_time = 0.0;
    
    // Warm-up run
    memset(C, 0, n * n * sizeof(double));
    func(A, B, C, n);
    
    // Timed runs
    for (int i = 0; i < iterations; i++) {
        memset(C, 0, n * n * sizeof(double));
        
        start = clock();
        func(A, B, C, n);
        end = clock();
        
        total_time += ((double)(end - start)) / CLOCKS_PER_SEC;
    }
    
    return total_time / iterations;
}