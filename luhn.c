#include <stdlib.h>
#include <stdio.h>

#include "gpu.h"

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

unsigned long unBitPackCC(unsigned long num){
	int i = 0;
	int digit = 0;
	unsigned long result = 0;
	for(i = 15; i >= 0; --i){
		digit = num & 15;
		printf("%d\n",digit);
		result << 4;
		result = result | digit;
		num = num >> 4;
	}
	return result;
}

int main(){
	// unsigned long start_point = 4425180000000000; //4,425,180,000,000,000
	unsigned long start_point = 4111111111111111;
	
	setupCUDA();
	
	int size = 1024;
	unsigned int *vector1 = (unsigned int*)malloc(sizeof(int) * size);
	unsigned int *vector2 = (unsigned int*)malloc(sizeof(int) * size);
	unsigned int *valid = (unsigned int*)malloc(sizeof(int) * size);
	
	int i = 0;
	int j = 0;
	// for(j = 0; j< 10240000; j++) {
	// 	unsigned long result = bitPackCC(start_point);
	// 	luhnOnPacked(result);
	// }
	
	for(j = 0; j < 1; j++)
	{
		for(i = 0; i < size; i++){
			unsigned long result = bitPackCC(start_point);
			// unsigned long result = 0;
			++start_point;
			unsigned int chunk1 = 0;
			unsigned int chunk2 = 0;
		
			chunk2 = result & 0xFFFFFFFF;
			result = result >> 32;
			chunk1 = result & 0xFFFFFFFF;
		
			vector1[i] = chunk1;
			vector2[i] = chunk2;
		}
	
		// unsigned int chunk1 = 0;
		// unsigned int chunk2 = 0;
		// 
		// chunk2 = result & 0xFFFFFFFF;
		// result = result >> 32;
		// chunk1 = result & 0xFFFFFFFF;
	
		test(vector1,vector2,valid);
	}
	// test(&chunk1,&chunk2);
	return 0;
}