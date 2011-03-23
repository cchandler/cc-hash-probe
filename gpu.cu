#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "gpu.h"

/*

High performance credit card hash probe using GPU.  Borrowed segments of Steve Worley's
SHA1 function from the EngineYard contest.

Thanks:
Steve Worley < m a t h g e e k@(my last name).com >
*/



#define SIZE     10000128

unsigned long long int *d_intervals;
	
unsigned int *d_start_positions_msd;
unsigned int *d_start_positions_lsd;

unsigned int *d_num1;
unsigned int *d_num2;

int* h_valid;
unsigned int* d_valid;
unsigned int* d_hash;

// __host__ __device__ unsigned int swapends(unsigned int v) 
// {
//   return 
//     ((255&(v>> 0))<<24)+
//     ((255&(v>> 8))<<16)+
//     ((255&(v>>16))<<8)+
//     ((255&(v>>24))<<0);
// }

/* 
   From Steve's notes:
   We don't want to precompute and store all 80 w array
   values. Instead we store only the next 16 values and update them in
   a logrolling array. Complicated but it means we can fit the tables
   in shared memory */
__device__ unsigned int popNextW(unsigned int *w, int &wIndex)
{
  unsigned int nextW=w[wIndex&15];
  int thisIndex=wIndex&15;
  w[thisIndex]^=w[(wIndex+16-3)&15]^w[(wIndex+16-8)&15]^w[(wIndex+16-14)&15];
  w[thisIndex]=  (w[thisIndex]<<1) | (w[thisIndex]>>31);
  ++wIndex;

  //  if (threadIdx.x==0) debugprint("pop %08x\n", nextW);
  return nextW;
}

__device__ unsigned int popFinalWs(unsigned int *w, int &wIndex)
{
  unsigned int nextW=w[wIndex&15];
  ++wIndex;
  return nextW;
}

__device__ int generateHash(unsigned int num1, unsigned int num2, unsigned int *hash){
	extern __shared__ unsigned int fullw[];
	
	int hash_offset = threadIdx.x + blockIdx.x * blockDim.x;
	
	unsigned int d_initVector[5];
	d_initVector[0] = 0x67452301;
	d_initVector[1] = 0xEFCDAB89;
	d_initVector[2] = 0x98BADCFE;
	d_initVector[3] = 0x10325476;
	d_initVector[4] = 0xC3D2E1F0;
	
	// unsigned int *w=fullw+17*threadIdx.x; // spaced by 17 to avoid bank conflicts, CC: TODO verify this
	char lookup_table[10] = {48,49,50,51,52,53,54,55,56,57};
	int pos = 0;
	unsigned int digit = 0;
	unsigned int num_1a = 0;
	unsigned int num_2a = 0;
	unsigned int num_3a = 0;
	unsigned int num_4a = 0;
	
	#pragma unroll 999
	for(pos = 0; pos <= 3; ++pos) {
		digit = 0;
		digit = num2 & (0xF << (pos * 4));
		digit = digit >> (pos * 4);
		
		num_1a = num_1a | lookup_table[digit];
		if(pos != 3) {num_1a = num_1a << 8;};
	}
	
	#pragma unroll 999
	for(pos = 4; pos <= 7; ++pos) {
		digit = 0;
		digit = num2 & (0xF << (pos * 4));
		digit = digit >> (pos * 4);
		
		num_2a = num_2a | lookup_table[digit];
		if(pos != 7) {num_2a = num_2a << 8;};
	}
	
	#pragma unroll 999
	for(pos = 0; pos <= 3; ++pos) {
		digit = 0;
		digit = num1 & (0xF << (pos * 4));
		digit = digit >> (pos * 4);
		
		num_3a = num_3a | lookup_table[digit];
		if(pos != 3) {num_3a = num_3a << 8;};
	}
	
	#pragma unroll 999
	for(pos = 4; pos <= 7; ++pos) {
		digit = 0;
		digit = num1 & (0xF << (pos * 4));
		digit = digit >> (pos * 4);
		
		num_4a = num_4a | lookup_table[digit];
		if(pos != 7) {num_4a = num_4a << 8;};
	}
	
	
	unsigned int w[80] = {'\0'};
	for (int i=0; i<80; i++) { w[i] = '\0'; };
	// w[0] = 1633837952; // 'abc' + 1 bit
	// num_1a = num_1a << 8;
	w[0] = num_1a;
	w[1] = num_2a;
	w[2] = num_3a;
	w[3] = num_4a;
	w[4] = (unsigned) 8 << 28;
	w[15] = 128;
	
	int wIndex=0;
	
	
	unsigned int a = d_initVector[0];
	unsigned int b = d_initVector[1];
    unsigned int c = d_initVector[2];
    unsigned int d = d_initVector[3];
    unsigned int e = d_initVector[4];
	
	#pragma unroll 999
	for (int i=0; i<20; ++i) {
	  unsigned int thisW=popNextW(w, wIndex);
	  // unsigned int thisW=w[i];
	  //    unsigned int f= (b&c)|((~b)&d);
	  unsigned int f= d ^ (b & (c^d)); // alternate computation of above
	  unsigned int temp=((a<<5)|(a>>27))+f+e+0x5A827999+thisW;
	  e=d;
	  d=c;
	  c=(b<<30)|(b>>2);
	  b=a;
	  a=temp;
	}
	
	#pragma unroll 999
	for (int i=20; i<40; ++i) {
	  unsigned int thisW=popNextW(w, wIndex);
	  // unsigned int thisW=w[i];
	  unsigned int f= b^c^d;
	  unsigned int temp=((a<<5)|(a>>27))+f+e+0x6ED9EBA1+thisW;
	  e=d;
	  d=c;
	  c=(b<<30)|(b>>2);
	  b=a;
	  a=temp;
	}

	#pragma unroll 999
	for (int i=40; i<60; ++i) {
	  unsigned int thisW=popNextW(w, wIndex);
	  // unsigned int thisW=w[i];
	  //    unsigned int f= (b&c) | (b&d) | (c&d);
	  unsigned int f= (b&c) | (d & (b|c)); // alternate computation of above
	  unsigned int temp=((a<<5)|(a>>27))+f+e+0x8F1BBCDC+thisW;
	  e=d;
	  d=c;
	  c=(b<<30)|(b>>2);
	  b=a;
	  a=temp;
	}

	#pragma unroll 999
	for (int i=60; i<64; ++i) {
	  unsigned int thisW=popNextW(w, wIndex);
	// unsigned int thisW=w[i];
	  unsigned int f= b^c^d;
	  unsigned int temp=((a<<5)|(a>>27))+f+e+0xCA62C1D6+thisW;
	  e=d;
	  d=c;
	  c=(b<<30)|(b>>2);
	  b=a;
	  a=temp;
	}


	#pragma unroll 999
	for (int i=64; i<80; ++i) {
	  unsigned int thisW=popFinalWs(w, wIndex); // simpler compute for final rounds
	  // unsigned int thisW=w[i];
	  unsigned int f= b^c^d;
	  unsigned int temp=((a<<5)|(a>>27))+f+e+0xCA62C1D6+thisW;
	  e=d;
	  d=c;
	  c=(b<<30)|(b>>2);
	  b=a;
	  a=temp;
	}
	
	hash[hash_offset*5 + 0] = a + d_initVector[0];
	hash[hash_offset*5 + 1] = b + d_initVector[1];
	hash[hash_offset*5 + 2] = c + d_initVector[2];
	hash[hash_offset*5 + 3] = d + d_initVector[3];
	hash[hash_offset*5 + 4] = e + d_initVector[4];
	
	return 0;
}

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

__global__ void GPUProbe(unsigned long long int *intervals, unsigned int *start_positions_msd,unsigned int *start_positions_lsd,unsigned int *num1, unsigned int *num2,unsigned int *valid, size_t valid_pitch, unsigned int *hash){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	
	unsigned long long int j = intervals[blockIdx.x] + threadIdx.x;
	j = GPUbitPackCC(j);
	
	unsigned int num_lsd = j;
	j = j >> 32;
	unsigned int num_msd = j;
	
	start_positions_lsd[threadIdx.x] = num_msd;
	
	unsigned int* valid_row = (unsigned int*)((char*)valid + 0 * valid_pitch); // 0 is the height offset. zero right now because it's essentially linear
	// unsigned int* num1_row = (unsigned int*)((char*)num1 + 0 * valid_pitch);
	// unsigned int* num2_row = (unsigned int*)((char*)num2 + 0 * valid_pitch);
	// 
	// num1_row[i] = num_msd;
	// num2_row[i] = num_lsd;
	
	
	// GPULuhn(&num1_row[i],&num2_row[i],&valid_row[i],valid_pitch); //TODO Remove dereference operator
	// generateHash(num1_row[i],num2_row[i],hash);
	
	GPULuhn(&num_msd,&num_lsd,&valid_row[i],valid_pitch);
	generateHash(num_msd,num_lsd,hash);
}

int setupCUDA(){
	int error = 0;
	int deviceCount = 0;
	error = cudaGetDeviceCount(&deviceCount);
	if (error != cudaSuccess) {
		printf("The system is reporting no devices available... %d\n",error);
		exit(-1);
	}
	
	error = cudaSetDevice(0);
	if(error != cudaSuccess){
		printf("Unable to set runtime device... %d\n",error);
		exit(-1);
	}
	
	return 0;
}

// extern "C"{
	int test(unsigned long *intervals, unsigned int *num1, unsigned int *num2,unsigned int *h_valid)
	{	
		int error = 0;
		
		// size_t d_start_positions_msd_pitch;
		// error = cudaMallocPitch((void **)(&d_start_positions_msd), &d_start_positions_msd_pitch, 192 * sizeof(int), 1); //This SIZE is the thread count
		// if(error != cudaSuccess){
		// 	printf("One of the mallocPitchs failed %d \n", error);
		// 	return -1;
		// }
		// size_t d_start_positions_lsd_pitch;
		// error = cudaMallocPitch((void **)(&d_start_positions_lsd), &d_start_positions_lsd_pitch, 192 * sizeof(int), 1); //This SIZE is the thread count
		// if(error != cudaSuccess){
		// 	printf("One of the mallocPitchs failed %d \n", error);
		// 	return -1;
		// }
		
		size_t d_intervals_pitch;
		error = cudaMallocPitch((void **)(&d_intervals), &d_intervals_pitch, blocksize * sizeof(int), 1); //This SIZE is the thread count
		if(error != cudaSuccess){
			printf("One of the mallocPitchs failed %d \n", error);
			return -1;
		}
		
		// size_t d_num1_pitch;
		// 		error = cudaMallocPitch((void **)(&d_num1), &d_num1_pitch, SIZE * sizeof(int), 1);
		// 		if(error != cudaSuccess){
		// 			printf("One of the mallocPitchs failed %d \n", error);
		// 			return -1;
		// 		}
		// 		size_t d_num2_pitch;
		// 		error = cudaMallocPitch((void **)(&d_num2), &d_num2_pitch, SIZE * sizeof(int), 1);
		// 		if(error != cudaSuccess){
		// 			printf("One of the mallocPitchs failed %d \n", error);
		// 			return -1;
		// 		}
		size_t d_valid_pitch;
		error = cudaMallocPitch((void **)(&d_valid), &d_valid_pitch, SIZE * sizeof(int), 1);
		if(error != cudaSuccess){
			printf("One of the mallocPitchs failed %d \n", error);
			return -1;
		}
		size_t d_hash_pitch;
		error = cudaMallocPitch((void **)(&d_hash), &d_hash_pitch, SIZE * sizeof(int) * 5, 1);
		
		if(error != cudaSuccess){
			printf("One of the mallocPitchs failed %d \n", error);
			return -1;
		}
		
		int *h_hash = (int*)malloc(SIZE * sizeof(int) * 5);
		unsigned int *h_start_positions_lsd = (unsigned int*)malloc(192 * sizeof(unsigned int));
		
	
		// error = cudaMemset2D(d_num1, d_num1_pitch, 0, SIZE * sizeof(int), 1);
		// 		error = cudaMemset2D(d_num2, d_num2_pitch, 0, SIZE * sizeof(int), 1);
		error = cudaMemset2D(d_valid, d_valid_pitch, 0, SIZE * sizeof(int), 1);
		error = cudaMemset2D(d_hash, d_hash_pitch, 0, SIZE * sizeof(int) * 5, 1);
		
		// error = cudaMemset2D(d_start_positions_msd, d_start_positions_msd_pitch, 0, 192 * sizeof(int), 1); //192 is the thread count
		// 		error = cudaMemset2D(d_start_positions_lsd, d_start_positions_lsd_pitch, 0, 192 * sizeof(int), 1); //192 is the thread count
		error = cudaMemset2D(d_intervals, d_intervals_pitch, 0, blocksize * sizeof(int), 1);
		
		// error = cudaMemcpy2D(d_num1,d_num1_pitch, num1, sizeof(int) * 512, 512 * sizeof(int), 1, cudaMemcpyHostToDevice);
		// error = cudaMemcpy2D(d_num2,d_num2_pitch, num2, sizeof(int) * 512, 512 * sizeof(int), 1, cudaMemcpyHostToDevice);
		error = cudaMemcpy2D(d_intervals,d_intervals_pitch, intervals, sizeof(int) * blocksize, blocksize * sizeof(int), 1, cudaMemcpyHostToDevice);
		
		if(error != cudaSuccess){
			printf("One of the mem copies or memsets failed %d\n", error);
			return -1;
		}
	
		GPUProbe<<< blocksize , threadsize >>>(d_intervals, d_start_positions_msd,d_start_positions_lsd, d_num1,d_num2,d_valid, d_valid_pitch, d_hash);
		cudaThreadSynchronize();
		
		
		// error = cudaMemcpy2D(num1, sizeof(int) * 512, d_num1, d_num1_pitch, 512 * sizeof(int), 1, cudaMemcpyDeviceToHost);
		// if(error != cudaSuccess){
		// 	printf("Failed to copy d_num1 from device %d\n", error);
		// }
		// error = cudaMemcpy2D(num2,sizeof(int) * 512, d_num2, d_num2_pitch, 512 * sizeof(int), 1, cudaMemcpyDeviceToHost);
		// if(error != cudaSuccess){
		// 	printf("Failed to copy d_num2 from device %d\n", error);
		// }
		
		// error = cudaMemcpy2D(h_start_positions_lsd, sizeof(int) * 192, d_start_positions_lsd, d_start_positions_lsd_pitch, 192 * sizeof(int), 1, cudaMemcpyDeviceToHost);
		// if(error != cudaSuccess){
		// 	printf("Failed to copy d_start_positions_lsd from device %d\n", error);
		// }
		
		error = cudaMemcpy2D(h_valid,sizeof(int) * 512, d_valid, d_valid_pitch, 512 * sizeof(int), 1, cudaMemcpyDeviceToHost);
		if(error != cudaSuccess){
			printf("Failed to copy d_valid from device %d\n",error);
		}
		
		error = cudaMemcpy2D(h_hash,sizeof(int) * 512 * 5, d_hash, d_hash_pitch, 512 * sizeof(int) * 5, 1, cudaMemcpyDeviceToHost);
		if(error != cudaSuccess){
			printf("Failed to copy d_hash from device %d\n",error);
			return -1;
		}
		
		if(error){
			printf("OMGWTFBBQ error %d\n",error);
			return -1;
		}
		// 
		// int i =0;
		// for(i = 0; i < 512; i++){
		// 	// if(h_valid[i]){
		// 		printf("%d --- Chunk1 %08x %08x  Valid %u\n",i, num2[i],num1[i], h_valid[i]);
		// 		printf("\tHash: %08x %08x %08x %08x %08x\n", h_hash[0 + i*5],h_hash[1 + i*5],h_hash[2 + i*5],h_hash[3 + i*5],h_hash[4 + i*5]);
		// 	// }
		// }
		// 
		// for(i = 0; i < 192; i++){
		// 	printf("%lu %u\n", intervals[i], h_start_positions_lsd[i]);
		// }
		// 
		// error = cudaFree(d_num1);
		// error = cudaFree(d_num2);
		error = cudaFree(d_valid);
		error = cudaFree(d_hash);
		// error = cudaFree(d_start_positions_msd);
		// error = cudaFree(d_start_positions_lsd);
		error = cudaFree(d_intervals);
		
		free(h_hash);
		free(h_start_positions_lsd);
		
		error = 0;
		if(error){
			printf("OMGWTFBBQ error %d\n",error);
			return -1;
		}
		
		// printf("Success...\n");
		
		return 0;
	}
// }
