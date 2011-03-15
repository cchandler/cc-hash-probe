#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define SIZE     1024
unsigned int *d_num1;
unsigned int *d_num2;

int* h_valid;
unsigned int* d_valid;
unsigned int* d_hash;


// __shared__ unsigned int initVector[5];
__device__ unsigned int d_initVector[5];

__host__ __device__ unsigned int swapends(unsigned int v) 
{
  return 
    ((255&(v>> 0))<<24)+
    ((255&(v>> 8))<<16)+
    ((255&(v>>16))<<8)+
    ((255&(v>>24))<<0);
}

/* We don't want to precompute and store all 80 w array
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

/* same as above but we don't need to compute more of the table  at the end. */
__device__ unsigned int popFinalWs(unsigned int *w, int &wIndex)
{
  unsigned int nextW=w[wIndex&15];
  ++wIndex;
  return nextW;
}

__device__ int generateHash(unsigned int *hash){
	extern __shared__ unsigned int fullw[];
	
	d_initVector[0] = 0x67452301;
	d_initVector[1] = 0xEFCDAB89;
	d_initVector[2] = 0x98BADCFE;
	d_initVector[3] = 0x10325476;
	d_initVector[4] = 0xC3D2E1F0;
	
	// unsigned int *w=fullw+17*threadIdx.x; // spaced by 17 to avoid bank conflicts, CC: I don't think this is relevant...
	// for (int i=0; i<16; ++i) w[i]='\0'; // Zeroing out the prep string
	// int wIndex=0;
	
	// unsigned int w[80]={   'a','a','a','a','a','a','a','a','a','a','a','a',
	//                     'a','a','a','a',0};
	
	// for (int i=0; i<16; ++i) w[i]='a';
	unsigned int w[80] = {'\0'};
	for (int i=0; i<80; i++) { w[i] = '\0'; };
	w[0] = 1633837952;
	w[15] = 24;
	
	int wIndex=0;
	
	// for (int i=0; i<16; i++) w[i]=swapends(w[i]);

	for (int i=16; i<80; i++) {
	  w[i]=w[i-3]^w[i-8]^w[i-14]^w[i-16];
	  w[i]=(w[i]<<1)|(w[i]>>31);
	}

	// unsigned int a = initVector[0];
	// unsigned int b = initVector[1];
	// unsigned int c = initVector[2];
	// unsigned int d = initVector[3];
	// unsigned int e = initVector[4];
	
	for(int j = 0; j < 1; j++){
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
	
	a = a + d_initVector[0];
	b = b + d_initVector[1];
	c = c + d_initVector[2];
	d = d + d_initVector[3];
	e = e + d_initVector[4];
	
	d_initVector[0] = a;
	d_initVector[1] = b;
	d_initVector[2] = c;
	d_initVector[3] = d;
	d_initVector[4] = e;
	}

	hash[0] = d_initVector[0];
	hash[1] = d_initVector[1];
	hash[2] = d_initVector[2];
	hash[3] = d_initVector[3];
	hash[4] = d_initVector[4];
	
	return 0;
}

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

__global__ void GPULuhn(unsigned int *num1, unsigned int *num2,unsigned int *valid, size_t valid_pitch, unsigned int *hash){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	// GPUbitPackCC(4111111111111111);
	// int i = 0;
	generateHash(hash);
	
	int pos = 0;
	unsigned int digit = 0;
	int even = 0;
	unsigned int sum = 0;
	int lookup_table[10] = {0,2,4,6,8,1,3,5,7,9};
	
	//Setup strided memory with correct pitch
	int* valid_row = (int*)((char*)valid + 0 * valid_pitch); // 0 is the height offset. zero right now because it's essentially linear
	int* num1_row = (int*)((char*)num1 + 0 * valid_pitch);
	int* num2_row = (int*)((char*)num2 + 0 * valid_pitch);
	
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
	
	valid_row[i] = (sum % 10 == 0);
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
		size_t d_num1_pitch;
		cudaMallocPitch((void **)(&d_num1), &d_num1_pitch, SIZE * sizeof(int), 1);
		size_t d_num2_pitch;
		cudaMallocPitch((void **)(&d_num2), &d_num2_pitch, SIZE * sizeof(int), 1);
		size_t d_valid_pitch;
		cudaMallocPitch((void **)(&d_valid), &d_valid_pitch, SIZE * sizeof(int), 1);
		size_t d_hash_pitch;
		cudaMallocPitch((void **)(&d_hash), &d_hash_pitch, SIZE * sizeof(int) * 5, 1);
		
		int *h_hash = (int*)malloc(SIZE * sizeof(int) * 5);
		
	
		cudaMemset2D(d_num1, d_num1_pitch, 0, SIZE * sizeof(int), 1);
		cudaMemset2D(d_num2, d_num2_pitch, 0, SIZE * sizeof(int), 1);
		cudaMemset2D(d_valid, d_valid_pitch, 0, SIZE * sizeof(int), 1);
		cudaMemset2D(d_hash, d_hash_pitch, 0, SIZE * sizeof(int) * 5, 1);
		
		cudaMemcpy2D(d_num1,d_num1_pitch, num1, sizeof(int) * SIZE, SIZE * sizeof(int), 1, cudaMemcpyHostToDevice);
		cudaMemcpy2D(d_num2,d_num2_pitch, num2, sizeof(int) * SIZE, SIZE * sizeof(int), 1, cudaMemcpyHostToDevice);
	
		GPULuhn<<< 1 , 1 >>>(d_num1,d_num2,d_valid, d_valid_pitch, d_hash);
		cudaThreadSynchronize();
		
		int error = 0;
		// error = cudaMemcpy2D(num1, sizeof(int) * SIZE, d_num1, d_num1_pitch, SIZE * sizeof(int), 1, cudaMemcpyDeviceToHost);
		if(error != cudaSuccess){
			printf("Failed to copy d_num1 from device %d\n", error);
		}
		
		// error = cudaMemcpy2D(num2,sizeof(int) * SIZE, d_num2, d_num2_pitch, SIZE * sizeof(int), 1, cudaMemcpyDeviceToHost);
		if(error != cudaSuccess){
			printf("Failed to copy d_num2 from device %d\n", error);
		}
		
		error = cudaMemcpy2D(h_valid,sizeof(int) * SIZE, d_valid, d_valid_pitch, SIZE * sizeof(int), 1, cudaMemcpyDeviceToHost);
		if(error != cudaSuccess){
			printf("Failed to copy d_valid from device %d\n",error);
		}
		
		error = cudaMemcpy2D(h_hash,sizeof(int) * SIZE * 5, d_hash, d_hash_pitch, SIZE * sizeof(int) * 5, 1, cudaMemcpyDeviceToHost);
		if(error != cudaSuccess){
			printf("Failed to copy d_hash from device %d\n",error);
		}
		
		if(error){
			printf("OMGWTFBBQ error %d\n",error);
		}

		int i =0;
		for(i = 0; i < 256; i++){
			if(h_valid[i]){
				printf("%d --- Chunk1 %u Chunk2 %u Valid %u\n",i, num1[i],num2[i], h_valid[i]);
				printf("\tHash: %08x %08x %08x %08x %08x\n", h_hash[0],h_hash[1],h_hash[2],h_hash[3],h_hash[4]);
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