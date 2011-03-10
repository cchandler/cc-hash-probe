#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define NUM_THREADS     4

typedef struct {
	unsigned long start_point;
	unsigned long end_point;
} cc_interval_t;

int luhn(char * num){
	int len = 15;
	int digit = 0;
	int even = 0;
	int sum = 0;
	
	for(len = 15; len >= 0; --len) {
		char c[] = {num[len], '\0'};
		digit = atoi(c);
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

long nextNumber(char * num) {
	long next_num = atol(num);
	next_num = next_num + 1;
	sprintf(num, "%ld", next_num);
	return next_num;
}

void *cc_probe(void *threadarg){
	cc_interval_t *mydata;
	mydata = (cc_interval_t *) threadarg;
	
	char val[20];
	sprintf(val, "%ld", mydata->start_point);
	
	unsigned long i = mydata->start_point;
	unsigned long j = 0;
	unsigned long total = 0;
	for(; i < mydata->end_point; i++) {
		++j;
		if(j % 100000000 == 0) {
			printf("processed %ld valid found %ld on %ld\n", j, total, i);
			break;
		}
		if(luhn(val)){
			++total;
		}
		nextNumber(val);
	}
	printf("Total valid #s:%ld\n",total);
	
	pthread_exit(NULL);
}

void divide_hash_space(unsigned long start, unsigned long end, cc_interval_t *data) {
	unsigned long total = end - start;
	unsigned long chunk = floor(total / NUM_THREADS);
	
	int i = 0;
	for(; i < NUM_THREADS; ++i){
		(data + i)->start_point = start + (chunk * i);
		(data + i)->end_point = start + ((chunk * (i + 1)) + 1);
	}
}

int main() {
	unsigned long start_point = 4425180000000000;
	unsigned long end_point = 4425190000000000;
	
	pthread_t threads[NUM_THREADS];
	cc_interval_t thread_data[NUM_THREADS];
	divide_hash_space(start_point,end_point,thread_data);
	
	long t;
	for(t = 0; t < NUM_THREADS; ++t) {
		int rc = pthread_create(&threads[t], NULL, cc_probe, (void *) &thread_data[t] );
	}
	
	pthread_exit(NULL);
	return 0;
}