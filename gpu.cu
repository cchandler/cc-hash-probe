/*
Copyright (C) 2011 by Chris Chandler

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN

*/

#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "gpu.h"
#include "util.h"


#include "gpu_sha_functions.cu"

/*
   Bitpack credit card number so we can work with it easily
*/
__device__ unsigned long long int GPUbitPackCC(unsigned long long int num){
	int i = 0;
	int digit = 0;
	unsigned long long int result = 0;
	for(i = 15; i >= 0; --i){
		digit = num % 10;
		num = num / 10;
		result = result << 4;
		result = result | digit;
	}
	return result;
}

/*
  Modified version of the Luhn check that only uses bit-shifts
*/
__device__ void GPULuhn(unsigned int *num1, unsigned int *num2,unsigned int *valid, size_t valid_pitch){
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
}

__global__ void GPUProbe(unsigned long long int *intervals, unsigned int *valid, size_t valid_pitch, unsigned int *hash){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	
	unsigned long long int j = intervals[blockIdx.x] + threadIdx.x;
	j = GPUbitPackCC(j);
	
	//Original implementation relied on 32 bit fields, so I still break them apart
	unsigned int num_lsd = j;
	j = j >> 32;
	unsigned int num_msd = j;
	
	unsigned int* valid_row = (unsigned int*)((char*)valid + 0 * valid_pitch); // 0 is the height offset. zero right now because we're using 1D memory
	
	GPULuhn(&num_msd,&num_lsd,&valid_row[i],valid_pitch);
	generateHash(num_msd,num_lsd,hash);
}

int getCudaDeviceCount(){
	int deviceCount = 0;
	int error = 0;
	error = cudaGetDeviceCount(&deviceCount);
	if (error != cudaSuccess) {
		printf("The system is reporting no devices available... %d\n",error);
		exit(-1);
	}
	return deviceCount;
}

int setupCUDA(cc_gpu_state_t *state){
	int error = 0;
	error = cudaSetDevice(state->gpuId);
	if(error != cudaSuccess){
		printf("Unable to set runtime device %d... %d\n",state->gpuId, error);
		exit(-1);
	}
	
	error = cudaMallocPitch((void **)(&(state->d_intervals)), &(state->d_intervals_pitch), get_block_size() * sizeof(long), 1);
	if(error != cudaSuccess){
		printf("Failed to allocate d_intervals_pitch %d \n", error);
		return -1;
	}
	
	error = cudaMallocPitch((void **)(&(state->d_valid)), &(state->d_valid_pitch), get_work_size() * sizeof(int), 1);
	if(error != cudaSuccess){
		printf("Failed to allocate d_valid_pitch %d \n", error);
		return -1;
	}
	
	error = cudaMallocPitch((void **)(&(state->d_hash)), &(state->d_hash_pitch), get_work_size() * sizeof(int) * HASH_CHUNKS, 1);
	if(error != cudaSuccess){
		printf("Failed to allocate d_hash_pitch %d \n", error);
		return -1;
	}
	
	return 0;
}

int teardownCUDA(cc_gpu_state_t *state){
	int error = 0;
	error = cudaFree(state->d_intervals);
	error = cudaFree(state->d_valid);
	error = cudaFree(state->d_hash);
	
	if(error != cudaSuccess){
		printf("Failed to free cuda memory %d \n", error);
		return -1;
	}
	
	return 0;
}

int cuda_scan(cc_gpu_state_t *state, unsigned long *intervals, unsigned int *h_valid, int *h_hashes)
{	
	int error = 0;
	
	if(error != cudaSuccess){
		printf("One of the mallocPitchs failed %d \n", error);
		return -1;
	}
	
	// error = cudaMemset2D(state->d_valid, state->d_valid_pitch, 0, SIZE * sizeof(int), 1);
	// error = cudaMemset2D(state->d_hash, state->d_hash_pitch, 0, SIZE * sizeof(int) * HASH_CHUNKS, 1);
	// error = cudaMemset2D(state->d_intervals, state->d_intervals_pitch, 0, blocksize * sizeof(int), 1);
	
	// Move interval data to the GPU
	error = cudaMemcpy2D(state->d_intervals,state->d_intervals_pitch, intervals, sizeof(int) * get_block_size(), get_block_size() * sizeof(int), 1, cudaMemcpyHostToDevice);
	if(error != cudaSuccess){
		printf("One of the mem copies or memsets failed %d\n", error);
		return -1;
	}

	// Execute the probe and wait for all threads to finish
	GPUProbe<<< get_block_size() , get_thread_size() >>>(state->d_intervals, state->d_valid, state->d_valid_pitch, state->d_hash);
	cudaThreadSynchronize();
	
	// Copy the Luhn validity data back to the host
	error = cudaMemcpy2D(h_valid,sizeof(int) * get_work_size(), state->d_valid, state->d_valid_pitch, get_work_size() * sizeof(int), 1, cudaMemcpyDeviceToHost);
	if(error != cudaSuccess){
		printf("Failed to copy d_valid from device %d\n",error);
	}
	// Copy the Hashes back to the host
	error = cudaMemcpy2D(h_hashes,sizeof(int) * get_work_size() * HASH_CHUNKS, state->d_hash, state->d_hash_pitch, get_work_size() * sizeof(int) * HASH_CHUNKS, 1, cudaMemcpyDeviceToHost);
	if(error != cudaSuccess){
		printf("Failed to copy d_hash from device %d\n",error);
		return -1;
	}
	
	// Display a sample of the first 5 records so we have something to look at
	int i =0;
	for(i = 0; i < 5; i++){
		// if(h_valid[i]){
			// printf("%d --- Chunk1 %08x %08x  Valid %u\n",i, num2[i],num1[i], h_valid[i]);
			printf("%d \tHash: %08x %08x %08x %08x %08x\n", i, h_hashes[0 + i*HASH_CHUNKS],h_hashes[1 + i*HASH_CHUNKS],h_hashes[2 + i*HASH_CHUNKS],h_hashes[3 + i*HASH_CHUNKS],h_hashes[4 + i*HASH_CHUNKS]);
		// }
	}
	
	return 0;
}
