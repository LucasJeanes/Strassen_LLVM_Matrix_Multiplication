#ifndef REFERENCE_DRIVER_H
#define REFERENCE_DRIVER_H

#include <stddef.h>

// Matrix utility functions
double* allocate_matrix(int n);
void free_matrix(double* matrix);
void print_matrix(const char* name, double* matrix, int n);

// Matrix initialization functions
void initialize_matrix_random(double* matrix, int n, double min_val, double max_val);
void initialize_matrix_identity(double* matrix, int n);
void initialize_matrix_sequential(double* matrix, int n);

// Matrix operations
void reference_multiply(double* A, double* B, double* C, int n);
int verify_matrices_equal(double* A, double* B, int n, double tolerance);

// Benchmarking
double benchmark_function(void (*func)(double*, double*, double*, int), 
                         double* A, double* B, double* C, int n, int iterations);

#endif // REFERENCE_DRIVER_H
