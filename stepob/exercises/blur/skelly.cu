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
  int h_va[N], h_vb[N], h_vc_cpu[N], h_vc_gpu[N];
  int *d_va, *d_vb, *d_vc;
  int i, ok;


  /*initialize vectors*/  
  for(i=0; i<N; i++){
    h_va[i] = i;
    h_vb[i] = N-i;    
  }
  
  /*call CPU function*/
  vsum(h_va, h_vb, h_vc_cpu, N);

  /*allocate memory on the GPU*/
  cudaMalloc(&d_va, N*sizeof(int));
  cudaMalloc(&d_vb, N*sizeof(int));
  cudaMalloc(&d_vc, N*sizeof(int));

  /*transmit data to GPU*/
  cudaMemcpy(d_va, h_va, N*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vb, h_vb, N*sizeof(int), cudaMemcpyHostToDevice);

  /*invoke the kernel on the GPU*/
  dim3 blocksPerGrid((N+BLOCKDIM-1)/BLOCKDIM, 1, 1);
  dim3 threadsPerBlock(BLOCKDIM, 1, 1);
  kernel<<<blocksPerGrid, threadsPerBlock>>>(d_va, d_vb, d_vc, N);

  /*transmit data from the GPU*/
  cudaMemcpy(h_vc_gpu, d_vc, N*sizeof(int), cudaMemcpyDeviceToHost);

  /*compare results*/
  for(i=0, ok=1; i<N && ok; i++)
    if(h_vc_cpu[i] != h_vc_gpu[i]){
      printf("Error!\n");
      ok = 0;
    }
  if(ok)
    printf("Success!\n");

  /*free memory*/
  cudaFree(d_va);
  cudaFree(d_vb);
  cudaFree(d_vc);
  
  return 0;
}

