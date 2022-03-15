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


//	LISA_search<index_t>::LcpInfo* s_info_one_d = (LISA_search<index_t>::LcpInfo*) aligned_alloc(64, sizeof(LISA_search<index_t>::LcpInfo) * 2 * pool_size);
	LcpInfo* s_info_one_d = (LcpInfo*) aligned_alloc(64, sizeof(LcpInfo) * 2 * pool_size);

	//s_info = (LISA_search<index_t>::LcpInfo**) aligned_alloc(64, sizeof(LISA_search<index_t>::LcpInfo*)*pool_size);
	s_info = (LcpInfo**) aligned_alloc(64, sizeof(LcpInfo*)*pool_size);
	for(int64_t i = 0; i < 2*pool_size; i = i + 2)
		s_info[i/2] = &s_info_one_d[i];

	s_msk = (uint8_t*) aligned_alloc(64, sizeof(uint8_t) * pool_size);

}

void threadData::dealloc_td(){

	free(chunk_pool);// = (Info*) aligned_alloc(64, sizeof(Info)*pool_size);
	free(str_enc);// = (uint64_t *)aligned_alloc(64, pool_size * sizeof(uint64_t));
	free(intv_all);// = (int64_t *)aligned_alloc(64, pool_size * 2 * sizeof(int64_t));
	free(fmi_pool);// = (Info*) aligned_alloc(64, sizeof(Info)*pool_size);

	free(tree_pool);// = (Info*) aligned_alloc(64, sizeof(Info)*pool_size);
	free(s_siz[0]);// = (index_t*) aligned_alloc(64, sizeof(index_t) * 2 * pool_size);

	free(s_siz);

	free(s_info[0]);// = (LISA_search<index_t>::LcpInfo*) aligned_alloc(64, sizeof(LISA_search<index_t>::LcpInfo) * 2 * pool_size);

	free(s_info);// = (LISA_search<index_t>::LcpInfo**) aligned_alloc(64, sizeof(LISA_search<index_t>::LcpInfo*)*pool_size);

	free(s_msk);// = (uint8_t*) aligned_alloc(64, sizeof(uint8_t) * pool_size);

}

