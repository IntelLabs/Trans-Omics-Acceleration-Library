#ifndef THREAD_DATA_H
#define THREAD_DATA_H
#ifdef VTUNE_ANALYSIS
#include <ittnotify.h>
#endif
#include"common.h"
class threadData {

	public:

		Info *chunk_pool;
		int chunk_cnt ;
		uint64_t *str_enc;
		int64_t *intv_all;
		
    		int64_t numSMEMs;
		Info *fmi_pool;
		int fmi_cnt;

		Info *tree_pool;
		int tree_cnt;
		index_t **s_siz;
		//LISA_search<index_t>::LcpInfo **s_info;
		LcpInfo **s_info;
		uint8_t *s_msk;
		threadData(int64_t pool_size);
		void dealloc_td();
};

#endif