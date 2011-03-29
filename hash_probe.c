#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <pthread.h>

#include <glib.h>

#include "gpu.h"
#include "cpu_func.h"
#include "util.h"

Config conf_data;
GHashTable *htable;

typedef struct{
	int threadId;
	int num_threads;
	int *hashes;
	unsigned int *valid;
	unsigned long table_size;
	unsigned long *intervals;
} scanjob;
	
void* scan_htable(void *threadarg){
	scanjob *mydata;
	mydata = (scanjob *) threadarg;
	
	unsigned long pos = mydata->threadId;
	int j =0;
	int *hashes = mydata->hashes;
	for(pos = mydata->threadId; pos < mydata->table_size; pos = pos + mydata->num_threads){
		if(mydata->valid[pos]){
			j++;
			char buffer[25];
			sprintf(buffer,"%08x%08x%08x%08x%08x", hashes[0 + pos*HASH_CHUNKS],hashes[1 + pos*HASH_CHUNKS],hashes[2 + pos*HASH_CHUNKS],hashes[3 + pos*HASH_CHUNKS],hashes[4 + pos*HASH_CHUNKS]);
			int * intval = (int*)g_hash_table_lookup(htable, (gpointer)buffer);
			if(intval){
				//Reverse the card # back out of the hash array position
				unsigned long block_loc = floor(pos / get_thread_size());
				unsigned long offset = pos % get_thread_size();
				unsigned long cc= mydata->intervals[block_loc] + offset;
				printf("Found a colission! %s %lu\n", buffer, cc);
			}
		}
	}
	
	pthread_exit(NULL);
	return 0;
}

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
			
			int thread_count = get_scan_thread_count();
			pthread_t threads[thread_count];
			scanjob thread_data[thread_count];
			
			int t = 0;
			for(t = 0; t < thread_count; ++t) {
				thread_data[t].threadId = t;
				thread_data[t].num_threads = thread_count;
				thread_data[t].hashes = hashes;
				thread_data[t].valid = valid;
				thread_data[t].table_size = get_work_size();
				thread_data[t].intervals = intervals;
				int rc = pthread_create(&threads[t], NULL, scan_htable, (void *) &thread_data[t] );
			}
			
			for(t = 0; t < thread_count; ++t){
				pthread_join(threads[t],NULL);
			}
			
			j++;
		}
		free(intervals);
		free(hashes);
		teardownCUDA(&(mydata->gpu_state));
	}else{
		unsigned int *hash = (unsigned int*)malloc(sizeof(int) * HASH_CHUNKS);
		unsigned long j = 0;
		
		while(start_point <= end_point){
			if(j % 1000000 == 0){
				printf("Processed %lu\n",j);
			}
			
			unsigned long result = cpu_bit_pack_CC(start_point);
			unsigned int valid = cpu_luhn_on_packed(result);
			unsigned int num1;
			unsigned int num2;
			num2 = 0xFFFFFFFF & result;
			result = result >> 32;
			num1 = 0xFFFFFFFF & result;
		
			cpu_sha1(num1,num2,hash);
			
			if(valid){
				char buffer[25];
				sprintf(buffer,"%08x%08x%08x%08x%08x", hash[0],hash[1],hash[2],hash[3],hash[4]);
				int * intval = (int*)g_hash_table_lookup(htable, (gpointer)buffer);
				if(intval){
					printf("Found a colission! %s %lu\n", buffer, start_point);
				}
				
			}
			++start_point;
			++j;
		}
	}
	
	printf("All possibilities between %lu and %lu have been processed\n", start_point, end_point);

	pthread_exit(NULL);
	return 0;
}

int main(){
	load_config();
	print_config();
	load_table(get_hash_file());
	
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