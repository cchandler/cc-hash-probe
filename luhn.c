#include <stdlib.h>
#include <stdio.h>

int test(unsigned int *num1, unsigned int *num2);

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
	unsigned long result = bitPackCC(start_point);
	
	int size = 1;
	unsigned int *vector1 = (unsigned int*)malloc(sizeof(int) * size);
	unsigned int *vector2 = (unsigned int*)malloc(sizeof(int) * size);
	
	// int i = 0;
	// for(i = 0; i < size; i++){
	// 	unsigned long result = bitPackCC(start_point);
	// 	++start_point;
	// 	unsigned int chunk1 = 0;
	// 	unsigned int chunk2 = 0;
	// 	
	// 	chunk2 = result & 0xFFFFFFFF;
	// 	result = result >> 32;
	// 	chunk1 = result & 0xFFFFFFFF;
	// 	
	// 	vector1[i] = chunk1;
	// 	vector2[i] = chunk2;
	// }
	
	unsigned int chunk1 = 0;
	unsigned int chunk2 = 0;

	// printf("Result %lu\n",result);
	chunk2 = result & 0xFFFFFFFF;
	// printf("Chunk 2 %d\n",chunk2);
	result = result >> 32;
	chunk1 = result & 0xFFFFFFFF;
	// printf("Chunk 1 %d\n",chunk1);
	
	// test(vector1,vector2);
	test(&chunk1,&chunk2);
	return 0;
}