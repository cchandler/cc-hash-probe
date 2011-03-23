#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>

#include "gpu.h"

#define SIZE     10000128

int regularGenerateHash(unsigned int num1, unsigned int num2, unsigned int *hash){
	unsigned int fullw[50];
	
	unsigned int d_initVector[5];
	d_initVector[0] = 0x67452301;
	d_initVector[1] = 0xEFCDAB89;
	d_initVector[2] = 0x98BADCFE;
	d_initVector[3] = 0x10325476;
	d_initVector[4] = 0xC3D2E1F0;
	
	// unsigned int *w=fullw+17*threadIdx.x; // spaced by 17 to avoid bank conflicts, CC: I don't think this is relevant...
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
	  // unsigned int thisW=popNextW(w, wIndex);
	  unsigned int thisW=w[i];
	     // unsigned int f= (b&c)|((~b)&d);
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
	  // unsigned int thisW=popNextW(w, wIndex);
	  unsigned int thisW=w[i];
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
	  // unsigned int thisW=popNextW(w, wIndex);
	  unsigned int thisW=w[i];
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
	  // unsigned int thisW=popNextW(w, wIndex);
	unsigned int thisW=w[i];
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
	  // unsigned int thisW=popFinalWs(w, wIndex); // simpler compute for final rounds
	  unsigned int thisW=w[i];
	  unsigned int f= b^c^d;
	  unsigned int temp=((a<<5)|(a>>27))+f+e+0xCA62C1D6+thisW;
	  e=d;
	  d=c;
	  c=(b<<30)|(b>>2);
	  b=a;
	  a=temp;
	}
	
	hash[0] = a + d_initVector[0];
	hash[1] = b + d_initVector[1];
	hash[2] = c + d_initVector[2];
	hash[3] = d + d_initVector[3];
	hash[4] = e + d_initVector[4];
	
	return 0;
}

unsigned int luhnOnPacked(unsigned long num){
	int len = 15;
	unsigned long digit = 0;
	int even = 0;
	unsigned long sum = 0;
	
	for(len = 15; len >= 0; --len) {
		digit = 0;
		digit = num & ((unsigned long)15 << (len * 4));
		digit = digit >> (len * 4);
		
		if(even) {
			digit = digit * 2;
			if(digit > 9){
				digit = digit - 9;
			}
		}
		sum = sum + digit;
		even = !even;
	}
	return (sum % 10 == 0);
}

unsigned long bitPackCC(unsigned long num){
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

// void bitPackCC_32(unsigned int num_msd, unsigned int num_lsd, unsigned int* output){
// 	int i = 0;
// 	unsigned int digit = 0;
// 	unsigned long result = 0;
// 	
// 	unsigned long unified = num_msd;
// 	unified = unified << 32;
// 	unified = unified | num_lsd;
// 	
// 	for(i = 15; i >= 0; --i){
// 		digit = unified % 10;
// 		unified = unified / 10;
// 		result = result << 4;
// 		result = result | digit;
// 	}
// 	
// 	output[1] = 0xFFFFFFFF & result;
// 	result = result >> 32;
// 	output[0] = 0xFFFFFFFF & result;
// }

// unsigned long unBitPackCC(unsigned long num){
// 	int i = 0;
// 	int digit = 0;
// 	unsigned long result = 0;
// 	for(i = 15; i >= 0; --i){
// 		digit = num & 15;
// 		printf("%d\n",digit);
// 		result << 4;
// 		result = result | digit;
// 		num = num >> 4;
// 	}
// 	return result;
// }

void incrementNumber(unsigned int *msd, unsigned int *lsd){
	unsigned int lsd_temp;
	lsd_temp = *lsd + 1;
	if(*lsd + 1 < lsd_temp){ //overflow!
		*msd = *msd + 1;
	}
	*lsd = lsd_temp;
}

unsigned long* divide_hash_space(unsigned long start, unsigned long end) {
	unsigned long total = end - start;
	unsigned long chunk = floor(total / blocksize);
	
	unsigned long *intervals = (unsigned long*)malloc(sizeof(long) * blocksize);
	int i = 0;
	for(; i < blocksize; ++i){
		intervals[i] = start + chunk * i;
	}
	return intervals;
}

int main(){
	//Setup the start points and ends points.
	//The longs are for easier host side computing and the msd/lsd WORDs are for
	//easier computing on the CUDA device.
	unsigned long start_point = 4111111111111111;
	unsigned long end_point =      4111999999999999;
	unsigned int start_point_msd = 957192;
	unsigned int start_point_lsd = 2775118279;
	unsigned int end_point_msd = 957399;
	unsigned int end_point_lsd = 2605776895;
	
#ifdef GPU
	setupCUDA();
#endif

	unsigned int *vector1 = (unsigned int*)malloc(sizeof(int) * SIZE);
	unsigned int *vector2 = (unsigned int*)malloc(sizeof(int) * SIZE);
	unsigned int *valid = (unsigned int*)malloc(sizeof(int) * SIZE);
	
	cc_struct temp;
	temp.unpacked0 = 0;
	temp.unpacked1 = 0;
	
	int i = 0;
	int j = 0;
	
#ifdef CPU
	// unsigned long start_point = 4111111111111111;
	unsigned int *hash = (unsigned int*)malloc(sizeof(int) * 5);
		for(j = 0; j< 100001280; j++) {
			unsigned long result = bitPackCC(start_point);
			luhnOnPacked(result);
			unsigned int num1;
			unsigned int num2;
			num2 = 0xFFFFFFFF & result;
			result = result >> 32;
			num1 = 0xFFFFFFFF & result;
			
			regularGenerateHash(num1,num2,hash);
		}
#endif
	
#ifdef GPU
	
	unsigned long* intervals = divide_hash_space(start_point,end_point);
	
	// unsigned int *packed = (unsigned int*) malloc(sizeof(unsigned int) * 3);
	// if(!packed){
	// 	printf("omgwtfbbq: how the fuck did malloc fail??\n");
	// 	exit(3);
	// }

	for(j = 0; j < 100; j++)
	{	
		
		// for(i = 0; i < SIZE; i++){
		// 	// unsigned long result = bitPackCC(start_point);
		// 	
		// 	bitPackCC_32(start_point_msd, start_point_lsd, packed);
		// 	unsigned long result = 0;
		// 	
		// 	unsigned long long_result = 0;
		// 	long_result = packed[0];
		// 	long_result = long_result << 32;
		// 	long_result = long_result | packed[1];
		// 	
		// 	unsigned long unified = start_point_msd;
		// 	unified = unified << 32;
		// 	unified = unified | start_point_lsd;
		// 	
		// 	// ++start_point;
		// 	while(!luhnOnPacked(long_result)){
		// 		incrementNumber(&start_point_msd, &start_point_lsd);
		// 		bitPackCC_32(start_point_msd, start_point_lsd, packed);
		// 		long_result = packed[0];
		// 		long_result = long_result << 32;
		// 		long_result = long_result | packed[1];
		// 		
		// 		unified = start_point_msd;
		// 		unified = unified << 32;
		// 		unified = unified | start_point_lsd;
		// 	}
		// 	
		// 	// printf("Valid! %u %u - %u %u - %lu - %lu\n", start_point_msd, start_point_lsd, packed[0], packed[1], long_result, unified);
		// 	// ++start_point;
		// 	incrementNumber(&start_point_msd, &start_point_lsd);
		// 
		// 	vector1[i] = packed[0];
		// 	vector2[i] = packed[1];	
		// 	
		// }
	
		test(intervals, vector1,vector2,valid);
	}
	// free(packed);
	
#endif

	// printf("Sleeping...\n");
	// sleep(3);

	return 0;
}