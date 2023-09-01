#ifndef THREAD_DATA_H
#define THREAD_DATA_H
#include"lisa_util.h"
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
		LcpInfo **s_info;
		uint8_t *s_msk;
		threadData(int64_t pool_size);
		void dealloc_td();
};

#endif
