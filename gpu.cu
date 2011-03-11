#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define SIZE     1
// long* h_num;
unsigned int *d_num1;
unsigned int *d_num2;

// int* h_numbers_seen;
// int* d_numbers_seen;

int* h_valid;
unsigned int* d_valid;

__global__ void GPULuhn(unsigned int *num1, unsigned int *num2,unsigned int *valid){
    // int i = threadIdx.x;
	// int i = 0;
	
	int pos = 0;
	unsigned int digit = 0;
	int even = 0;
	unsigned int sum = 0;
	int lookup_table[10] = {0,2,4,6,8,1,3,5,7,9};
	
	
	for(pos = 7; pos >= 0; --pos) {
		digit = 0;
		digit = *num1 & (0xF << (pos * 4));
		digit = digit >> (pos * 4);
		
		if(even) {
			digit = lookup_table[digit];
		}
		sum = sum + digit;
		even = !even;
	}
	for(pos = 7; pos >= 0; --pos) {
		digit = 0;
		digit = *num2 & (0xF << (pos * 4));
		digit = digit >> (pos * 4);
		
		if(even) {
			digit = lookup_table[digit];
		}
		sum = sum + digit;
		even = !even;
	}
	
	
	*valid = (sum % 10 == 0);
	// *valid = num1[0];
	// valid[i] = (sum % 10 == 0);
	// valid[i] = sum;
}

// extern "C"{
	int test(unsigned int *num1, unsigned int *num2)
	{
		// h_num = (long*)malloc(sizeof(long));
		h_valid = (int*)malloc(sizeof(int) * SIZE);
		// h_numbers_seen = (int*)malloc(sizeof(int) * 15);
		
		cudaMalloc((void**)&d_num1,sizeof(int) * SIZE);
		cudaMalloc((void**)&d_num2,sizeof(int) * SIZE);
		cudaMalloc((void**)&d_valid,sizeof(int) * SIZE);
		int i = 0;
		
		// for(i = 0; i < SIZE; i++){
		// 	h_valid[i] = 0;
		// }
		
		// cudaMalloc((void**)&d_numbers_seen,sizeof(int)*15);
	
	
		// cudaMemset	(d_num1, 0, sizeof(int) * SIZE);
		// cudaMemset	(d_num2, 0, sizeof(int) * SIZE);
		// cudaMemset	(d_valid, 0, sizeof(int) * SIZE);
		cudaMemcpy(d_num1,num1, sizeof(int) * SIZE, cudaMemcpyHostToDevice);
		cudaMemcpy(d_num2,num2, sizeof(int) * SIZE, cudaMemcpyHostToDevice);
	
		GPULuhn<<< 1, 1 >>>(d_num1,d_num2,d_valid);
		cudaThreadSynchronize();
		// printf("Final %ld\n",*h_num);
		cudaMemcpy(h_valid, d_valid, sizeof(int) * SIZE, cudaMemcpyDeviceToHost);
		
		for(i = 0; i < SIZE; i++){
			// if(h_valid[i]){
				printf("Chunk1 %u Chunk2 %u Valid %u\n",*num1,*num2, *h_valid);
			// }
		}
		
		return 0;
	}
// }