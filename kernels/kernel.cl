/*
 * vector_add.cl
 * OpenCL C kernel for element-wise addition of two integer vectors.
 */

__kernel void vector_add(__global const int *A,
                         __global const int *B,
                         __global int *C,
                         int num_elements)
{
    // Get the unique global index for this work-item
    int i = get_global_id(0);

    // Perform the element-wise addition
    // Note: In a real-world scenario with arbitrary sizes,
    // you might add a bounds check: if (i < vector_size) { ... }
    // where vector_size is passed as another kernel argument.
    // For this basic example, we assume global_size matches vector size.
    if(i<num_elements){
      C[i] = A[i] + B[i];
    }
}
