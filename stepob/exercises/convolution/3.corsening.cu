#include <stdio.h>
#include <cuda_runtime.h>

#define MATRIX_DIM 512
#define FILTER_RADIUS 32
#define FILTER_DIM ((FILTER_RADIUS * 2) + 1)
#define BLOCKDIM 2
#define MAX_VALUE 3
#define THREADS_PER_BLOCK 32
#define GRID_DIM 4

#define CHECK(call)                                                                       \
    {                                                                                     \
        const cudaError_t err = call;                                                     \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

#define CHECK_KERNELCALL()                                                                \
    {                                                                                     \
        const cudaError_t err = cudaGetLastError();                                       \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

__constant__ int* filter[FILTER_DIM*FILTER_DIM];

void print_matrix(int* m, int dim){
    for(int i=0;i<dim;i++){
        for(int j=0;j<dim;j++){
            printf("%d ",m[i*dim+j]);
        }
        printf("\n");
    }
    printf("\n");
}

void convolution_cpu(int *input,
                     const int *filter,
                     int *output,
                     const int width,
                     const int height,
                     const int filter_size,
                     const int filter_radius)
{
  for (int outRow = 0; outRow < width; outRow++)
  {
    for (int outCol = 0; outCol < height; outCol++)
    {
      int value=0;
      for (int row = 0; row < filter_size; row++)
        for (int col = 0; col < filter_size; col++)
        {
          int inRow = outRow - filter_radius + row;
          int inCol = outCol - filter_radius + col;
          if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
          {
            value += filter[row * filter_size + col] * input[inRow * width + inCol];
          }
        }
      output[outRow * width + outCol] = value;
    }
  }
}

/*GPU kernel performing vector addition c = a + b*/
__global__ void kernel(const int *input, const int *filter, int *res)
{
    int thread_row = blockDim.y*blockIdx.y + threadIdx.y;
    int thread_column = blockDim.x*blockIdx.x + threadIdx.x;
    float acc = 0;

        for(int stride_row=0, i=0; i<(MATRIX_DIM/(THREADS_PER_BLOCK*GRID_DIM)); stride_row+=THREADS_PER_BLOCK*GRID_DIM, i++)
            for(int stride_col=0, j=0; j<(MATRIX_DIM/(THREADS_PER_BLOCK*GRID_DIM)); stride_col+=THREADS_PER_BLOCK*GRID_DIM, j++){
                acc =0;
                for(int row=0; row<FILTER_DIM; row++){
                    for(int col=0; col<FILTER_DIM; col++){

                        int input_row = thread_row - FILTER_RADIUS + row + stride_row;
                        int input_column = thread_column - FILTER_RADIUS + col + stride_col;
                        if(input_row>=0 && input_row<MATRIX_DIM && input_column>=0 && input_column<MATRIX_DIM)
                            acc += input[input_row*MATRIX_DIM + input_column] * filter[row*FILTER_DIM + col];
                    }
                } 
                res[(thread_row+stride_row)*MATRIX_DIM + thread_column + stride_col] = acc;      
            }  
    
}

int main()
{
    int h_input_matrix[MATRIX_DIM * MATRIX_DIM], h_filter[FILTER_DIM * FILTER_DIM];
    int h_res_cpu[MATRIX_DIM * MATRIX_DIM], h_res_gpu[MATRIX_DIM * MATRIX_DIM];
    int *d_input, *d_filter, *d_res;
    int i, ok;

    /*initialize vectors*/
    srand(0);
    for(i=0; i<MATRIX_DIM*MATRIX_DIM; i++)
        h_input_matrix[i] = rand()%MAX_VALUE + 1;
    for(i=0; i<FILTER_DIM*FILTER_DIM; i++)
        h_filter[i] = rand()%MAX_VALUE + 1;

    /*call CPU function*/
    convolution_cpu(h_input_matrix,h_filter,h_res_cpu,MATRIX_DIM,MATRIX_DIM,FILTER_DIM,FILTER_RADIUS);

    printf("Matrixex\n");
    print_matrix(h_input_matrix,MATRIX_DIM);
    print_matrix(h_filter,FILTER_DIM);

    printf("CPU res\n");
    print_matrix(h_res_cpu,MATRIX_DIM);
    

    /*allocate memory on the GPU*/
    CHECK(cudaMalloc(&d_input, MATRIX_DIM * MATRIX_DIM * sizeof(int)))
    CHECK(cudaMalloc(&d_filter, FILTER_DIM * FILTER_DIM * sizeof(int)))
    CHECK(cudaMalloc(&d_res, MATRIX_DIM * MATRIX_DIM * sizeof(int)))

    /*transmit data to GPU*/
    CHECK(cudaMemcpy(d_input, h_input_matrix, MATRIX_DIM * MATRIX_DIM * sizeof(int), cudaMemcpyHostToDevice))
    CHECK(cudaMemcpy(d_filter, h_filter, FILTER_DIM * FILTER_DIM * sizeof(int), cudaMemcpyHostToDevice))

    /*invoke the kernel on the GPU*/
    dim3 blocksPerGrid(GRID_DIM, GRID_DIM, 1);
    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    CHECK(cudaMemcpyToSymbol(filter,h_filter,FILTER_DIM*FILTER_DIM,0,cudaMemcpyHostToDevice))
    kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_filter, d_res);
    CHECK_KERNELCALL()

    /*transmit data from the GPU*/
    CHECK(cudaMemcpy(h_res_gpu, d_res, MATRIX_DIM * MATRIX_DIM * sizeof(int), cudaMemcpyDeviceToHost))

    printf("GPU res\n");
    print_matrix(h_res_gpu,MATRIX_DIM);


    /*compare results*/
    for (i=0, ok=1; i<MATRIX_DIM*MATRIX_DIM && ok; i++)
        if (h_res_cpu[i] != h_res_gpu[i])
        {
            printf("Error!\n");
            ok = 0;
        }
    if (ok)
        printf("Success!\n");

    /*free memory*/
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_res);

    return 0;
}
