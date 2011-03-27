#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <pthread.h>

#include "gpu.h"

int regularGenerateHash(unsigned int num1, unsigned int num2, unsigned int *hash){
	
	unsigned int d_initVector[5];
	d_initVector[0] = 0x67452301;
	d_initVector[1] = 0xEFCDAB89;
	d_initVector[2] = 0x98BADCFE;
	d_initVector[3] = 0x10325476;
	d_initVector[4] = 0xC3D2E1F0;
	
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
	  unsigned int thisW=w[i];
	  unsigned int f= d ^ (b & (c^d));
	  unsigned int temp=((a<<5)|(a>>27))+f+e+0x5A827999+thisW;
	  e=d;
	  d=c;
	  c=(b<<30)|(b>>2);
	  b=a;
	  a=temp;
	}
	
	#pragma unroll 999
	for (int i=20; i<40; ++i) {
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
	  unsigned int thisW=w[i];
	  unsigned int f= (b&c) | (d & (b|c));
	  unsigned int temp=((a<<5)|(a>>27))+f+e+0x8F1BBCDC+thisW;
	  e=d;
	  d=c;
	  c=(b<<30)|(b>>2);
	  b=a;
	  a=temp;
	}

	#pragma unroll 999
	for (int i=60; i<64; ++i) {
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

int divide_hash_space_for_gpu(unsigned long start, unsigned long end, unsigned long offset, unsigned long *intervals) {
	unsigned long total = end - start;
	unsigned long chunk = floor(total / blocksize);
	
	if(chunk < offset){ // If this is the case we've scanned the entire range
		return 0;
	}
	
	int i = 0;
	for(; i < blocksize; ++i){
		intervals[i] = start + (chunk * i + offset);
	}
	return 1;
}

void divide_hash_space_for_threads(unsigned long start, unsigned long end, cc_interval_t *data, int thread_count) {
	unsigned long total = end - start;
	unsigned long chunk = ceil(total / thread_count);
	
	int i = 0;
	for(; i < thread_count; ++i){
		(data + i)->start_point = start + (chunk * i);
		(data + i)->end_point = start + ((chunk * (i + 1)) + 1);
	}
}

void* process_work(void *threadarg){
	cc_interval_t *mydata;
	mydata = (cc_interval_t *) threadarg;
	
	printf("Activating thread ID %d!\n", mydata->threadId);
	printf("Handling interval: %lu - %lu\n", mydata->start_point, mydata->end_point);
	unsigned long start_point = mydata->start_point;
	unsigned long end_point =   mydata->end_point;
	
#ifdef GPU
	setupCUDA(&(mydata->gpu_state));
#endif
	
	int i = 0;
	int j = 0;
	
#ifdef CPU
	unsigned int *hash = (unsigned int*)malloc(sizeof(int) * HASH_CHUNKS);
	while(start_point <= end_point){
		unsigned long result = bitPackCC(start_point);
		luhnOnPacked(result);
		unsigned int num1;
		unsigned int num2;
		num2 = 0xFFFFFFFF & result;
		result = result >> 32;
		num1 = 0xFFFFFFFF & result;
		
		regularGenerateHash(num1,num2,hash);
		start_point++;
	}
#endif
	
#ifdef GPU

	unsigned int *valid = (unsigned int*)malloc(sizeof(int) * SIZE);	
	unsigned long *intervals = (unsigned long*)malloc(sizeof(long) * blocksize);
	int *hashes = (int*)malloc(SIZE * sizeof(int) * HASH_CHUNKS);
	
	int more_work = 1;

	while( divide_hash_space_for_gpu(start_point,end_point, j * threadsize, intervals) )
	{	
		if(j % 10 == 0){
			printf("Processed %lu\n",j * SIZE);
		}
	
		cuda_scan(&(mydata->gpu_state),intervals,valid, hashes);
		j++;
	}
	free(intervals);
	free(hashes);
	teardownCUDA(&(mydata->gpu_state));
#endif

	
	printf("All possibilities between %lu and %lu have been processed\n", start_point, end_point);

	pthread_exit(NULL);
	return 0;
}


#ifdef CPU
int getThreadCount(){
	return 4;
}
#endif

#ifdef GPU
int getThreadCount(){
	return getCudaDeviceCount();
}
#endif

int main(){
	unsigned long start_point = 4461577000000000;
	// unsigned long start_point = 4461577999900000;
	unsigned long end_point =   4461577999999999;
	
	int thread_count = getThreadCount();
	pthread_t threads[thread_count];
	cc_interval_t thread_data[thread_count];
	
	printf("Initializing system with %d thread(s)\n", thread_count);
	printf("Executing in mode: %s\n",MODE);
	printf("Scan start point: %lu\n", start_point);
	printf("Scan end point: %lu\n",end_point);
	
	divide_hash_space_for_threads(start_point,end_point,thread_data, thread_count);
	
	int t = 0;
	for(t = 0; t < thread_count; ++t) {
		thread_data[t].threadId = t;
#ifdef GPU
		thread_data[t].gpu_state.gpuId = t;
#endif
		int rc = pthread_create(&threads[t], NULL, process_work, (void *) &thread_data[t] );
	}
	
	for(t = 0; t < thread_count; ++t){
		pthread_join(threads[t],NULL);
	}
	
	return 0;
}