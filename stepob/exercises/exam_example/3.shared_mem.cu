#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCKDIM 32
#define MAXVAL 100
#define DIST 10

void printV(int *V, int num);
void compute(int *V, int *R, int num);

//display a vector of numbers on the screen
void printV(int *V, int num) {
  int i;
  for(i=0; i<num; i++)
      printf("%3d(%d) \n", V[i], i);
  printf("\n");    
}

//kernel function: identify peaks in the vector
void compute(int *V, int *R, int num) {
  int i, j, ok;
  for(i=0; i<num; i++){
    for(j=-DIST, ok=1; j<=DIST; j++){
      if(i+j>=0 && i+j<num && j!=0 && V[i]<=V[i+j])
        ok=0;
    }
    R[i] = ok;
  }
}

/*GPU kernel performing vector addition c = a + b*/
__global__ void kernel(int *input, int *res, int dim)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int shared_array[BLOCKDIM+(DIST*2)];
    int my_element = 0, tmp = 1;

  shared_array[DIST+threadIdx.x] = input[blockIdx.x*blockDim.x+threadIdx.x];
  if(threadIdx.x<DIST){
    if(blockIdx.x>0)
      shared_array[threadIdx.x] = input[blockIdx.x*blockDim.x+threadIdx.x-DIST];
    else
      shared_array[threadIdx.x] = 0;
  }
  if(threadIdx.x>=blockDim.x-DIST){
    if(blockIdx.x<gridDim.x-1)
      shared_array[threadIdx.x+DIST*2] = input[(blockIdx.x)*blockDim.x+threadIdx.x+DIST];
    else
      shared_array[DIST*2+threadIdx.x] = 0;  
  }

    __syncthreads();
    if(index<dim && index>=0){
        my_element = input[index];
        for(int i=-DIST; i<=DIST; i++){
            if(my_element<=shared_array[i+threadIdx.x+DIST] && i!=0)
                tmp=0;
        }
        res[index] = tmp;
    }
    

}

int main(int argc, char** argv)
{
    int* h_va, *h_res_cpu, *h_res_gpu;
    int *d_va, *d_res;
    int i, dim, ok;

    if(argc!=2){
    printf("Please specify sizes of the input vector\n");
    return 0;
    }

    dim=atoi(argv[1]);

    h_va = (int*) malloc(sizeof(int) * dim);
    if(!h_va){
        printf("Error: malloc failed\n");
        return 1;
    }
    h_res_cpu = (int*) malloc(sizeof(int) * dim);
    if(!h_res_cpu){
        printf("Error: malloc failed\n");
        return 1;
    }
    h_res_gpu = (int*) malloc(sizeof(int) * dim);
    if(!h_res_gpu){
        printf("Error: malloc failed\n");
        return 1;
    }

    //initialize input vectors
    srand(0);
    for(i=0; i<dim; i++)
    h_va[i] = rand()%MAXVAL +1;

    /*call CPU function*/
    compute(h_va, h_res_cpu, dim);
 
    //print results
    printf("results on the cpu\n");
    //printV(h_va, dim);
    //printV(h_res_cpu, dim);
    

    /*allocate memory on the GPU*/
    cudaMalloc(&d_va, dim * sizeof(int));
    cudaMalloc(&d_res, dim * sizeof(int));

    /*transmit data to GPU*/
    cudaMemcpy(d_va, h_va, dim * sizeof(int), cudaMemcpyHostToDevice);

    /*invoke the kernel on the GPU*/
    dim3 blocksPerGrid((dim + BLOCKDIM - 1) / BLOCKDIM, 1, 1);
    dim3 threadsPerBlock(BLOCKDIM, 1, 1);
    kernel<<<blocksPerGrid, threadsPerBlock>>>(d_va, d_res, dim);

    /*transmit data from the GPU*/
    cudaMemcpy(h_res_gpu, d_res, dim * sizeof(int), cudaMemcpyDeviceToHost);

    //print results
    printf("results on the GPU\n");
    //printV(h_va, dim);
    //printV(h_res_gpu, dim);

      /*compare results*/
    for(i=0, ok=1; i<dim && ok; i++)
        if(h_res_cpu[i] != h_res_gpu[i]){
            printf("Error! index %d\n",i);
            ok = 0;
        }
    if(ok)
        printf("Success!\n");


    /*free memory*/
    cudaFree(d_va);
    cudaFree(d_res);
    free(h_va);
    free(h_res_cpu);
    free(h_res_gpu);


    return 0;
}
