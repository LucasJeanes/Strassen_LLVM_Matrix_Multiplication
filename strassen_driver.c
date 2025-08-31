#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "reference_driver.h"

// Forward declarations for LLVM IR functions
extern void strassen_multiply(double* A, double* B, double* C, int n);

int main(int argc, char* argv[]) {
    int n = 128; // Default matrix size
    int test_correctness = 1;
    int run_benchmark = 1;
    int print_matrices = 0;
    int benchmark_iterations = 5;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "-print") == 0) {
            print_matrices = 1;
        } else if (strcmp(argv[i], "-no-test") == 0) {
            test_correctness = 0;
        } else if (strcmp(argv[i], "-no-benchmark") == 0) {
            run_benchmark = 0;
        } else if (strcmp(argv[i], "-iterations") == 0 && i + 1 < argc) {
            benchmark_iterations = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "-help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  -n <size>        Matrix size (must be power of 2, default: 128)\n");
            printf("  -print           Print matrices (only for small sizes)\n");
            printf("  -no-test         Skip correctness testing\n");
            printf("  -no-benchmark    Skip benchmark testing\n");
            printf("  -iterations <n>  Number of benchmark iterations (default: 5)\n");
            printf("  -help, -h        Show this help message\n");
            return 0;
        }
    }
    
    // Validate matrix size (must be power of 2 for Strassen)
    // n = 8 : 1000, n-1 = 7 : 0111
    // 8 & 7 : 1000 & 0111 => 0000 (= n^2)
    if (n <= 0 || (n & (n - 1)) != 0) {
        printf("Error: Matrix size must be a positive power of 2\n");
        return 1;
    }
    
    printf("=== Strassen Matrix Multiplication Test ===\n");
    printf("Matrix size: %d x %d\n", n, n);
    printf("Memory usage: %.2f MB per matrix\n", (n * n * sizeof(double)) / (1024.0 * 1024.0));
    
    // Allocate matrices
    double* A = allocate_matrix(n);
    double* B = allocate_matrix(n);
    double* C_strassen = allocate_matrix(n);
    double* C_reference = allocate_matrix(n);
    
    if (!A || !B || !C_strassen || !C_reference) {
        printf("Error: Failed to allocate memory for matrices\n");
        return 1;
    }
    
    // Initialize matrices with random values
    srand((unsigned int)time(NULL));
    printf("\nInitializing matrices with random values...\n");
    initialize_matrix_random(A, n, -10.0, 10.0);
    initialize_matrix_random(B, n, -10.0, 10.0);
    
    // Print matrices if requested and size is small
    if (print_matrices && n <= 8) {
        print_matrix("Matrix A", A, n);
        print_matrix("Matrix B", B, n);
    }
    
    // Test correctness
    if (test_correctness) {
        printf("\n=== Correctness Testing ===\n");
        
        // Clear result matrices
        memset(C_strassen, 0, n * n * sizeof(double));
        memset(C_reference, 0, n * n * sizeof(double));
        
        printf("Computing reference result (standard algorithm)...\n");
        reference_multiply(A, B, C_reference, n);
        
        printf("Computing Strassen result...\n");
        strassen_multiply(A, B, C_strassen, n);
        
        // Verify results (due to potential rounding errors of floats)
        double tolerance = 1e-10;
        int strassen_correct = verify_matrices_equal(C_strassen, C_reference, n, tolerance);
        
        printf("\nResults:\n");
        printf("Strassen vs Reference: %s\n", strassen_correct ? "PASS" : "FAIL");
        
        if (print_matrices && n <= 8) {
            print_matrix("Reference Result", C_reference, n);
            print_matrix("Strassen Result", C_strassen, n);
        }
        
        if (!strassen_correct) {
            printf("Error: Correctness test failed!\n");
            // Don't exit, continue with benchmark if requested
        }
    }
    
    // Benchmark performance
    if (run_benchmark) {
        printf("\n=== Performance Benchmark ===\n");
        printf("Running %d iterations each...\n\n", benchmark_iterations);
        
        double time_reference = benchmark_function(reference_multiply, A, B, C_reference, n, benchmark_iterations);
        double time_strassen = benchmark_function(strassen_multiply, A, B, C_strassen, n, benchmark_iterations);
        
        // Calculate GFLOPS (n^3 * 2 operations for matrix multiplication)
        double operations = 2.0 * n * n * n;
        double gflops_reference = operations / (time_reference * 1e9);
        double gflops_strassen = operations / (time_strassen * 1e9);
        
        printf("Performance Results:\n");
        printf("Reference C:    %8.3f ms  (%6.2f GFLOPS)\n", time_reference * 1000, gflops_reference);
        printf("Strassen LLVM:  %8.3f ms  (%6.2f GFLOPS)  [%.2fx vs Reference]\n", 
               time_strassen * 1000, gflops_strassen, time_reference / time_strassen);
        
        printf("\nSpeedup Analysis:\n");
        printf("Strassen vs Reference: %.2fx\n", time_reference / time_strassen);
        
        // Expected complexity analysis
        double expected_ratio = pow(n, 3) / pow(n, log2(7));
        printf("Theoretical Strassen advantage: %.2fx (for large n)\n", expected_ratio);
    }
    
    // Test with different matrix patterns
    if (test_correctness && n <= 64) {
        printf("\n=== Additional Pattern Tests ===\n");
        
        // Identity matrix test
        initialize_matrix_identity(A, n);
        initialize_matrix_random(B, n, -5.0, 5.0);
        
        memset(C_strassen, 0, n * n * sizeof(double));
        memset(C_reference, 0, n * n * sizeof(double));
        
        reference_multiply(A, B, C_reference, n);
        strassen_multiply(A, B, C_strassen, n);
        
        int identity_test = verify_matrices_equal(C_strassen, C_reference, n, 1e-10);
        printf("Identity matrix test: %s\n", identity_test ? "PASS" : "FAIL");
        
        // Sequential matrix test
        initialize_matrix_sequential(A, n);
        initialize_matrix_sequential(B, n);
        
        memset(C_strassen, 0, n * n * sizeof(double));
        memset(C_reference, 0, n * n * sizeof(double));
        
        reference_multiply(A, B, C_reference, n);
        strassen_multiply(A, B, C_strassen, n);
        
        int sequential_test = verify_matrices_equal(C_strassen, C_reference, n, 1e-9);
        printf("Sequential matrix test: %s\n", sequential_test ? "PASS" : "FAIL");
    }
    
    // Clean up
    free_matrix(A);
    free_matrix(B);
    free_matrix(C_strassen);
    free_matrix(C_reference);
    
    printf("\nTest completed successfully!\n");
    return 0;
}