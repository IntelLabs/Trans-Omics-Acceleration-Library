#include "thread_data.h"

threadData::threadData(int64_t pool_size){

	numSMEMs = 0;
	chunk_pool = (Info*) aligned_alloc(64, sizeof(Info)*pool_size);
	chunk_cnt = 0;
	str_enc = (uint64_t *)aligned_alloc(64, pool_size * sizeof(uint64_t));
	intv_all = (int64_t *)aligned_alloc(64, pool_size * 2 * sizeof(int64_t));
	fmi_pool = (Info*) aligned_alloc(64, sizeof(Info)*pool_size);
	fmi_cnt = 0;

	tree_pool = (Info*) aligned_alloc(64, sizeof(Info)*pool_size);
	tree_cnt = 0;
	index_t* s_siz_one_d = (index_t*) aligned_alloc(64, sizeof(index_t) * 2 * pool_size);
	s_siz = (index_t**) aligned_alloc(64, sizeof(index_t*) * pool_size);
	for(int64_t i = 0; i < 2*pool_size; i = i + 2)
		s_siz[i/2] = &s_siz_one_d[i];


	LcpInfo* s_info_one_d = (LcpInfo*) aligned_alloc(64, sizeof(LcpInfo) * 2 * pool_size);

	s_info = (LcpInfo**) aligned_alloc(64, sizeof(LcpInfo*)*pool_size);
	for(int64_t i = 0; i < 2*pool_size; i = i + 2)
		s_info[i/2] = &s_info_one_d[i];

	s_msk = (uint8_t*) aligned_alloc(64, sizeof(uint8_t) * pool_size);

}

void threadData::dealloc_td(){
	free(chunk_pool);
	free(str_enc);
	free(intv_all);
	free(fmi_pool);
	free(tree_pool);
	free(s_siz[0]);
	free(s_siz);
	free(s_info[0]);
	free(s_info);
	free(s_msk);
}

