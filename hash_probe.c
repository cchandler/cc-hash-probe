#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <pthread.h>

#include <glib.h>

#include "gpu.h"
#include "util.h"

Config conf_data;

void* process_work(void *threadarg){
	cc_interval_t *mydata;
	mydata = (cc_interval_t *) threadarg;
	
	printf("Activating thread ID %d!\n", mydata->threadId);
	printf("Handling interval: %lu - %lu\n", mydata->start_point, mydata->end_point);
	unsigned long start_point = mydata->start_point;
	unsigned long end_point =   mydata->end_point;
	
	int i = 0;
	int j = 0;
	
	if(is_gpu_mode()){
		setupCUDA(&(mydata->gpu_state));
		
		unsigned int *valid = (unsigned int*)malloc(sizeof(int) * get_work_size());	
		unsigned long *intervals = (unsigned long*)malloc(sizeof(long) * get_block_size());
		int *hashes = (int*)malloc(get_work_size() * sizeof(int) * HASH_CHUNKS);

		int more_work = 1;

		while( divide_hash_space_for_gpu(start_point,end_point, j * get_thread_size(), intervals) )
		{	
			if(j % 10 == 0){
				printf("Processed %lu\n",j * get_work_size());
			}

			cuda_scan(&(mydata->gpu_state),intervals,valid, hashes);
			j++;
		}
		free(intervals);
		free(hashes);
		teardownCUDA(&(mydata->gpu_state));
	}else{
		// unsigned int *hash = (unsigned int*)malloc(sizeof(int) * HASH_CHUNKS);
		// while(start_point <= end_point){
		// 	unsigned long result = bitPackCC(start_point);
		// 	luhnOnPacked(result);
		// 	unsigned int num1;
		// 	unsigned int num2;
		// 	num2 = 0xFFFFFFFF & result;
		// 	result = result >> 32;
		// 	num1 = 0xFFFFFFFF & result;
		// 
		// 	regularGenerateHash(num1,num2,hash);
		// 	start_point++;
		// }
	}
	
	printf("All possibilities between %lu and %lu have been processed\n", start_point, end_point);

	pthread_exit(NULL);
	return 0;
}

int main(){
	load_config();
	print_config();
	
	int thread_count = get_hash_thread_count();
	pthread_t threads[thread_count];
	cc_interval_t thread_data[thread_count];
	
	divide_hash_space_for_threads(get_cc_start_point(),get_cc_end_point(),thread_data, thread_count);
	
	int t = 0;
	for(t = 0; t < thread_count; ++t) {
		thread_data[t].threadId = t;
		thread_data[t].gpu_state.gpuId = t;
		int rc = pthread_create(&threads[t], NULL, process_work, (void *) &thread_data[t] );
	}
	
	for(t = 0; t < thread_count; ++t){
		pthread_join(threads[t],NULL);
	}
	
	return 0;
}