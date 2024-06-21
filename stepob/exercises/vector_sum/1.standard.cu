#include<stdio.h>
#include<cuda_runtime.h>

#define N (1<<16)
#define BLOCKDIM 64

void vsum(int* a, int* b, int* c, int dim);
__global__ void kernel(int* a, int* b, int* c, int dim);


/*CPU function performing vector addition c = a + b*/
void vsum(int* a, int* b, int* c, int dim){
  int i;
  for(i=0; i<dim; i++)
    c[i] = a[i] + b[i];
}

/*GPU kernel performing vector addition c = a + b*/
__global__ void kernel(int* a, int* b, int* c, int dim){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<dim)
    c[i] = a[i] + b[i]; 
}

int main(){
  int host_va[N],host_vb[N],host_res_CPU[N],host_res_GPU[N];
  int *d_va, *d_vb, *d_res;
  int i, ok;


  /*initialize vectors*/  
  for(i=0; i<N; i++){
    host_va[i] = i;
    host_vb[i] = N-i;    
  }
  
  /*call CPU function*/
  vsum(host_va, host_vb, host_res_CPU, N);

  /*allocate memory on the GPU*/
  cudaMalloc(&d_va, N*sizeof(int));
  cudaMalloc(&d_vb, N*sizeof(int));
  cudaMalloc(&d_res, N*sizeof(int));

  /*transmit data to GPU*/
  cudaMemcpy(d_va, host_va, N*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vb, host_vb, N*sizeof(int), cudaMemcpyHostToDevice);

  /*invoke the kernel on the GPU*/
  dim3 blocksPerGrid((N+BLOCKDIM-1)/BLOCKDIM, 1, 1);
  dim3 threadsPerBlock(BLOCKDIM, 1, 1);
  kernel<<<blocksPerGrid, threadsPerBlock>>>(d_va, d_vb, d_res, N);

  /*transmit data from the GPU*/
  cudaMemcpy(host_res_GPU, d_res, N*sizeof(int), cudaMemcpyDeviceToHost);

  /*compare results*/
  for(i=0, ok=1; i<N && ok; i++)
    if(host_res_CPU[i] != host_res_GPU[i]){
      printf("Error!\n");
      ok = 0;
    }
  if(ok)
    printf("Success!\n");

  /*free memory*/
  cudaFree(d_va);
  cudaFree(d_vb);
  cudaFree(d_res);
  
  return 0;
}




