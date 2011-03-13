#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define SIZE     1024
// long* h_num;
unsigned int *d_num1;
unsigned int *d_num2;

// int* h_numbers_seen;
// int* d_numbers_seen;

int* h_valid;
unsigned int* d_valid;

__device__ unsigned long GPUbitPackCC(unsigned long num){
	int i = 0;
	int digit = 0;
	unsigned long result = 0;
	for(i = 15; i >= 0; --i){
		digit = num % 10;
		num = num / 10;
		result = result << 4;
		result = result | digit;
	}
	return result;
}

__global__ void GPULuhn(unsigned int *num1, unsigned int *num2,unsigned int *valid, size_t valid_pitch){
	int i = threadIdx.x;
	// GPUbitPackCC(4111111);
	// int i = 0;
	
	int pos = 0;
	unsigned int digit = 0;
	int even = 0;
	unsigned int sum = 0;
	int lookup_table[10] = {0,2,4,6,8,1,3,5,7,9};
	
	//Setup strided memory with correct pitch
	int* valid_row = (int*)((char*)valid + 1 * valid_pitch); // 1 is the "height"
	int* num1_row = (int*)((char*)num1 + 1 * valid_pitch);
	int* num2_row = (int*)((char*)num2 + 1 * valid_pitch);
	
	
	for(pos = 7; pos >= 0; --pos) {
		digit = 0;
		digit = num1_row[i] & (0xF << (pos * 4));
		digit = digit >> (pos * 4);
		
		if(even) {
			digit = lookup_table[digit];
		}
		sum = sum + digit;
		even = !even;
	}
	for(pos = 7; pos >= 0; --pos) {
		digit = 0;
		digit = num2_row[i] & (0xF << (pos * 4));
		digit = digit >> (pos * 4);
		
		if(even) {
			digit = lookup_table[digit];
		}
		sum = sum + digit;
		even = !even;
	}
	
	
	// valid[i] = (sum % 10 == 0);
	valid_row[i] = (sum % 10 == 0);
	// valid[i] = num2[i];
	// *(valid + i * stride) = 15;
}

int setupCUDA(){
	int error = 0;
	int deviceCount = 0;
	error = cudaGetDeviceCount(&deviceCount);
	if (error != cudaSuccess) {
		printf("The system is reporting no devices available... %d\n",error);
	}
	
	error = cudaSetDevice(0);
	if(error != cudaSuccess){
		printf("Unable to set runtime device... %d\n",error);
	}
	
	return 0;
}

// extern "C"{
	int test(unsigned int *num1, unsigned int *num2,unsigned int *h_valid)
	{
		// h_num = (long*)malloc(sizeof(long));
		// h_numbers_seen = (int*)malloc(sizeof(int) * 15);
		
		size_t d_num1_stride;
		cudaMallocPitch((void **)(&d_num1), &d_num1_stride, SIZE * sizeof(int), 1);
		// cudaMalloc((void**)&d_num1,sizeof(int) * SIZE);
		
		size_t d_num2_stride;
		cudaMallocPitch((void **)(&d_num2), &d_num2_stride, SIZE * sizeof(int), 1);
		// cudaMalloc((void**)&d_num2,sizeof(int) * SIZE);
		
		size_t d_valid_pitch;
		cudaMallocPitch((void **)(&d_valid), &d_valid_pitch, SIZE * sizeof(int), 1);
		// cudaMalloc((void**)&d_valid,sizeof(int) * SIZE);

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
	
		GPULuhn<<< 1 , 512 >>>(d_num1,d_num2,d_valid, d_valid_pitch);
		cudaThreadSynchronize();
		cudaMemcpy(h_valid, d_valid, sizeof(int) * SIZE, cudaMemcpyDeviceToHost);
		
		int error = 0;
		if(error){
			printf("OMGWTFBBQ error %d\n",error);
		}

		// int i =0;
		for(i = 0; i < 10; i++){
			if(h_valid[i]){
				printf("Chunk1 %u Chunk2 %u Valid %u\n",num1[i],num2[i], h_valid[i]);
				// printf("Chunk1 %u Chunk2 %u Valid %u\n",num1[i],num2[i], *(h_valid + i * d_valid_stride));
			}
		}
		
		cudaFree(&d_num1);
		cudaFree(&d_num2);
		cudaFree(&d_valid);
		
		error = 0;
		if(error){
			printf("OMGWTFBBQ error %d\n",error);
		}
		
		return 0;
	}
// }