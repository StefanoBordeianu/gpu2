/*
Vector addition.
* Version 0: the sum is performed by a function on the CPU
*/

#include<stdio.h>
#define N (1<<16)

void vsum(int* a, int* b, int* c, int dim);

/*CPU function performing vector addition c = a + b*/
void vsum(int* a, int* b, int* c, int dim){
  int i;
  for(i=0; i<dim; i++)
    c[i] = a[i] + b[i];
}

int main(){
  int h_va[N], h_vb[N], h_vc[N];
  int i;

  /*initialize vectors*/  
  for(i=0; i<N; i++){
    h_va[i] = i;
    h_vb[i] = N-i;    
  }
  
  /*call CPU function*/
  vsum(h_va, h_vb, h_vc, N);
  
  /*we don't print the results...*/
  printf("Done!\n");
  
  return 0;
}



