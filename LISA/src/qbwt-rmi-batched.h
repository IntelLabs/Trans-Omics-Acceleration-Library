/*************************************************************************************
MIT License

Copyright (c) 2020 Intel Labs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Authors: Saurabh Kalikar <saurabh.kalikar@intel.com>; Sanchit Misra <sanchit.misra@intel.com>
*****************************************************************************************/
#ifdef VTUNE_ANALYSIS
#include <ittnotify.h>
#endif
#include "common.h"
#include <cmath>
#include "read.h"
#include <immintrin.h>
#include <fstream>
#include <omp.h>
int K;
constexpr int L = 1;
constexpr double alphas[L] = {2.1};
#ifdef ENABLE_PREFETCH
#define my_prefetch(a, b) _mm_prefetch(a, b)
#else
#define my_prefetch(a, b)
#endif 
//constexpr int B = _B;
    uint64_t sumSMEMLength = 0;
    int64_t numSMEMs = 0;
    int64_t sorting_ticks = 0;
    int64_t one_ticks = 0;
    int64_t chunk_ticks = 0;
    int64_t partial_chunk_ticks = 0;
    int64_t prune_succ = 0;
    int64_t one_calls = 0;
//int64_t chc = 0, pc = 0, oc = 0;
int64_t chc = 0, chc_success = 0, chk_failed_proned = 0, chc_failed = 0, pc = 0, oc = 0, largeLcp = 0;
int64_t totalTicks = 0;
//#include "qbwt-hybrid-lut.h"
#include "qbwt-ipbwt-rmi.h"
//#include "bitMap.h"
#include "chunkEncode.h"



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
		QBWT_HYBRID_LUT<index_t>::LcpInfo **s_info;
		uint8_t *s_msk;
/*		
		threadData(){
		//	chunk_pool = NULL;
			chunk_cnt = 0;
		//	str_enc = NULL;
		//	intv_all = NULL;
			
			numSMEMs = 0;
		//	fmi_pool = NULL;
		 	fmi_cnt = 0;

		//	tree_pool = NULL;
			tree_cnt = 0;
			s_siz = 0;
		//	s_info = NULL;
		//	s_msk = NULL;
		}



		~threadData(){
			if(str_enc != NULL)
				free(str_enc);
			if(intv_all != NULL)
				free(intv_all);
			if(this->s_msk != NULL)
				free(this->s_msk);

		}
*/	
		threadData(int64_t pool_size){

    			numSMEMs = 0;
			chunk_pool = (Info*) aligned_alloc(64, sizeof(Info)*pool_size);
			chunk_cnt = 0;
			str_enc = (uint64_t *)aligned_alloc(64, pool_size * sizeof(uint64_t));
			intv_all = (int64_t *)aligned_alloc(64, pool_size * 2 * sizeof(int64_t));
//			str_enc = (uint64_t *)malloc(pool_size * sizeof(uint64_t));
//			intv_all = (int64_t *)malloc(pool_size * 2 * sizeof(int64_t));
			//-----------------------------------------------//
			fmi_pool = (Info*) aligned_alloc(64, sizeof(Info)*pool_size);
			fmi_cnt = 0;
			//-----------------------------------------------//

			tree_pool = (Info*) aligned_alloc(64, sizeof(Info)*pool_size);
			tree_cnt = 0;
			index_t* s_siz_one_d = (index_t*) aligned_alloc(64, sizeof(index_t) * 2 * pool_size);
			s_siz = (index_t**) aligned_alloc(64, sizeof(index_t*) * pool_size);
			for(int64_t i = 0; i < 2*pool_size; i = i + 2)
				s_siz[i/2] = &s_siz_one_d[i];


			QBWT_HYBRID_LUT<index_t>::LcpInfo* s_info_one_d = (QBWT_HYBRID_LUT<index_t>::LcpInfo*) aligned_alloc(64, sizeof(QBWT_HYBRID_LUT<index_t>::LcpInfo) * 2 * pool_size);

			s_info = (QBWT_HYBRID_LUT<index_t>::LcpInfo**) aligned_alloc(64, sizeof(QBWT_HYBRID_LUT<index_t>::LcpInfo*)*pool_size);
			for(int64_t i = 0; i < 2*pool_size; i = i + 2)
				s_info[i/2] = &s_info_one_d[i];

//			s_msk = (uint8_t*) malloc(sizeof(uint8_t) * pool_size);
			s_msk = (uint8_t*) aligned_alloc(64, sizeof(uint8_t) * pool_size);

		}


};


#define S_SWP_END do{\
        int c; \
	if(cnt > pref_dist)\
		c = --cnt;\
	else\
		c = --shrink_batch_size;\
        q = tree_pool[c]; \
        siz[0] = s_siz[c][0]; siz[1] = s_siz[c][1]; \
        info[0] = s_info[c][0]; info[1] = s_info[c][1]; \
        msk = s_msk[c]; \
}while(0)

#define S_RUN do{\
    bool small = siz[1] < siz[0]; \
    if(info[small].bw_ext_msk & msk) { \
        if(small) { \
            q.r = q.l + qbwt.get_lcp(q.intv.second); \
            q.intv.second = q.intv.first + siz[1]; \
        } else { \
            q.r = q.l + qbwt.get_lcp(q.intv.first); \
            q.intv.first = q.intv.second - siz[0]; \
        } \
	fmi_pool[fmi_cnt++] = q;\
        S_SWP_END;\
    } else { \
        if(small) { \
            q.intv.second = q.intv.first + siz[1]; \
            info[1] = qbwt.lcpi[q.intv.second]; \
            siz[1] = GET_WIDTH(info[1].s_width, q.intv.second); \
        } else { \
            q.intv.first = q.intv.second - siz[0]; \
            info[0] = qbwt.lcpi[q.intv.first]; \
            siz[0] = GET_WIDTH(info[0].s_width, q.intv.first); \
        } \
    }\ 
}while(0)


#define S_PREFETCH do{\
            bool small = siz[1] < siz[0]; \
                my_prefetch((const char*)(qbwt.lcpp1 + (small ? q.intv.second : q.intv.first)), _MM_HINT_T0); \
                my_prefetch((const char*)(qbwt.lcpi + (small ?  \
                                q.intv.first + siz[1] : q.intv.second - siz[0])), _MM_HINT_T0); \
}while(0)

#define GET_WIDTH(A, B) (A == qbwt.WID_MAX ? \
        (index_t)lower_bound(qbwt.b_width.begin(), qbwt.b_width.end(), B,  \
            [&](pair<index_t, index_t> p, index_t qq){return p.first < qq;})->second: \
            (index_t)A)


#define S_LOAD(i) \
            Info &q = tree_pool[i]; \
            index_t* siz = s_siz[i]; \
            QBWT_HYBRID_LUT<index_t>::LcpInfo* info = s_info[i]; \
            uint8_t &msk = s_msk[i] 

inline void s_pb(QBWT_HYBRID_LUT<index_t> &qbwt, Info &_q, int cnt, threadData &td) {
//    auto &cnt = tree_cnt;

//    S_LOAD(cnt);
    Info &q = td.tree_pool[cnt]; 
    index_t* siz = td.s_siz[cnt]; 
    QBWT_HYBRID_LUT<index_t>::LcpInfo* info = td.s_info[cnt]; 
    uint8_t &msk = td.s_msk[cnt]; 
    q = _q;
    info[0] = qbwt.lcpi[q.intv.first]; info[1] = qbwt.lcpi[q.intv.second];
    siz[0] = GET_WIDTH(info[0].s_width, q.intv.first); siz[1] = GET_WIDTH(info[1].s_width, q.intv.second);
#ifndef NO_DNA_ORD
    msk = 1<<dna_ord(q.p[q.l-1]);
#else 
    msk = 1<<q.p[q.l-1];
#endif 
//    cnt++;
}



void tree_shrink_batched( QBWT_HYBRID_LUT<index_t> &qbwt, int cnt, threadData &td){

	Info* tree_pool = td.tree_pool;
	//int &tree_cnt = td.tree_cnt;
	index_t** s_siz = td.s_siz;
	QBWT_HYBRID_LUT<index_t>::LcpInfo** s_info = td.s_info;
	uint8_t* s_msk = td.s_msk;

	Info* fmi_pool = td.fmi_pool;//[pool_size];
	int &fmi_cnt = td.fmi_cnt;




	int pref_dist = 50;
//	Info shrink_batch[50];
	int shrink_batch_size = pref_dist = min(pref_dist, cnt);
	pref_dist = shrink_batch_size;

	
	while(shrink_batch_size > 0) {
		for(int i = 0; i < shrink_batch_size; i++){
			    S_LOAD(i);
			    S_RUN;
			    S_PREFETCH;
		}
	}
}



void fmi_extend_batched( QBWT_HYBRID_LUT<index_t> &qbwt, int cnt, Info* q_batch, threadData &td, Output* output, int min_seed_len){

	Info* tree_pool = td.tree_pool;
	int &tree_cnt = td.tree_cnt;
	
	int pref_dist = 30;
	int fmi_batch_size = pref_dist = min(pref_dist, cnt);
	pref_dist = fmi_batch_size;

	Info pf_batch[fmi_batch_size];
	
	auto cnt1 = fmi_batch_size;

	// prepare first batch
	for(int i = 0; i < fmi_batch_size; i++){
		Info &q = q_batch[i];	
		static constexpr int INDEX_T_BITS = sizeof(index_t)*__CHAR_BIT__;
		static constexpr int shift = __lg(INDEX_T_BITS);
		auto ls = ((q.intv.first>>shift)<<3), hs = ((q.intv.second>>shift)<<3); 
		my_prefetch((const char*)(qbwt.fmi.occb + ls), _MM_HINT_T0); 
		my_prefetch((const char*)(qbwt.fmi.occb + hs), _MM_HINT_T0); 
                my_prefetch((const char*)(q.p + q.l -  1) , _MM_HINT_T0);//cache 
	}
	
	while(fmi_batch_size > 0) {

		for(int i = 0; i < fmi_batch_size; i++){
			Info &q = q_batch[i];	
			//process one step
			int it = q.l -1;

			if(it >=0){			
				auto next = qbwt.fmi.backward_extend({q.intv.first, q.intv.second}, q.p[it]);
				if(next.low >= next.high) { 
				
					tree_pool[tree_cnt++] = q;  //State change

					if(q.r - q.l >= min_seed_len){
#ifdef OUTPUT
/*						output[q.id].qPos.push_back({q.l, q.r});
						output[q.id].refPos.push_back(q.intv);
						output[q.id].smem.push_back(SMEM_out(q.id, q.l, q.r, q.intv.first, q.intv.second));
*/
						output->smem[td.numSMEMs++] = SMEM_out(q.id, q.l, q.r, q.intv.first, q.intv.second);                                                                                                            			
#endif
					}
					if(cnt1< cnt)// > pref_dist) //More queries to be processed?
						q = q_batch[cnt1++]; //direction +
					else
						q = q_batch[--fmi_batch_size];
          			      	my_prefetch((const char*)(q.p + q.l -  1) , _MM_HINT_T0);

				}
				else {
					q.l = it, q.intv={next.low, next.high};  //fmi-continue
				}
			}	
			else{
					//query finished
			//		q_batch[q.id] = q;


					if(q.r - q.l >= min_seed_len){
#ifdef OUTPUT
//						output[q.id].qPos.push_back({q.l, q.r});
//						output[q.id].refPos.push_back(q.intv);
//						output[q.id].smem.push_back(SMEM_out(q.id, q.l, q.r, q.intv.first, q.intv.second));

						output->smem[td.numSMEMs++] = SMEM_out(q.id, q.l, q.r, q.intv.first, q.intv.second);                                                                                                            			
#endif
					}
					if(cnt1 < cnt)// > pref_dist) //More queries to be processed?
						q = q_batch[cnt1++];
					else
						q = q_batch[--fmi_batch_size];
                			my_prefetch((const char*)(q.p + q.l -  1) , _MM_HINT_T0);
			}
			static constexpr int INDEX_T_BITS = sizeof(index_t)*__CHAR_BIT__;
			static constexpr int shift = __lg(INDEX_T_BITS);
			auto ls = ((q.intv.first>>shift)<<3), hs = ((q.intv.second>>shift)<<3); 
			my_prefetch((const char*)(qbwt.fmi.occb + ls + 4), _MM_HINT_T0); 
			my_prefetch((const char*)(qbwt.fmi.occb + hs + 4), _MM_HINT_T0); 
		}
	}
}

void smem_rmi_batched(Info *qs, int64_t qs_size, int64_t batch_size, QBWT_HYBRID_LUT<index_t> &qbwt, threadData &td, Output* output, int min_seed_len){
		


	Info *chunk_pool = td.chunk_pool;//(Info*) aligned_alloc(64, sizeof(Info)*pool_size);
	int &chunk_cnt = td.chunk_cnt;
	uint64_t *str_enc = td.str_enc;//(uint64_t *)malloc(pool_size * sizeof(uint64_t));
	int64_t *intv_all = td.intv_all;//(int64_t *)malloc(pool_size * 2 * sizeof(int64_t));
	//-----------------------------------------------//
	Info* fmi_pool = td.fmi_pool;//[pool_size];
	int &fmi_cnt = td.fmi_cnt;
	//-----------------------------------------------//

	Info* tree_pool = td.tree_pool;//[pool_size];
	int &tree_cnt = td.tree_cnt;
	//index_t** s_siz = td.s_siz;//[pool_size][2];
	//QBWT_HYBRID_LUT<index_t>::LcpInfo** s_info = td.s_info;//[pool_size][2];
	//uint8_t* s_msk = td.s_msk;//[pool_size];
	int64_t next_q = 0;
	while(next_q < qs_size || (chunk_cnt + fmi_cnt ) > 0){
		while(next_q < qs_size && chunk_cnt < batch_size && fmi_cnt < batch_size){
				if(qs[next_q].r >= K){
					chunk_pool[chunk_cnt++] = qs[next_q++];
				}
				else
					fmi_pool[fmi_cnt++] = qs[next_q++];
		}
		
		// process chunk batch
		if(next_q >= qs_size || !(chunk_cnt < batch_size)){



//		prepareChunkBatchVectorized_v1(chunk_pool, chunk_cnt, str_enc, intv_all);	
		prepareChunkBatch(chunk_pool, chunk_cnt, str_enc, intv_all);



		qbwt.rmi.backward_extend_chunk_batched(&str_enc[0], chunk_cnt, intv_all);
		//qbwt.rmi.backward_extend_chunk_batched(&chunk_pool[0], chunk_cnt, intv_all);
			 
		    
                    auto cnt = chunk_cnt;
		    chunk_cnt = 0;
		    for(int64_t j = 0; j < cnt; j++) {
			Info &q = chunk_pool[j];
			auto next_intv = make_pair(intv_all[2*j],intv_all[2*j + 1]);

			// next state: chunk batching
			if(next_intv.first < next_intv.second){
				//q.numPrevSuccChk++;
				q.intv = next_intv;
				q.l -= K;
			
				if(q.l >= K)// && !(q.intv.second - q.intv.first > 1 && q.intv.second - q.intv.first < 100))	
					chunk_pool[chunk_cnt++] = q;
				else 
				if(q.l > 0){
					fmi_pool[fmi_cnt++] = q;
				}
				else if(q.r - q.l >= min_seed_len){
			
#ifdef OUTPUT
//					output[q.id].qPos.push_back({q.l, q.r});
//					output[q.id].refPos.push_back(q.intv);
					//output[q.id].smem.push_back(SMEM_out(q.id, q.l, q.r, q.intv.first, q.intv.second));
			

					output->smem[td.numSMEMs++] = SMEM_out(q.id, q.l, q.r, q.intv.first, q.intv.second);                                                                                                            			
							
#endif
				}
			}
			else {
			//Next state: fmi procssing
				fmi_pool[fmi_cnt++] = q;
			}
			
		    }
		}
		
		// fmi processing
		if(next_q >= qs_size || !(fmi_cnt < batch_size)){
		    auto cnt = fmi_cnt;
		    fmi_cnt = 0;	

		    fmi_extend_batched(qbwt, cnt, &fmi_pool[0], td, output, min_seed_len);	

		    cnt = tree_cnt;
		    tree_cnt = 0;
		    for(int i = 0; i < cnt; i++) {
			auto &q = tree_pool[i];
			s_pb(qbwt, q, i, td);
                        my_prefetch((const char*)(qbwt.lcpi + tree_pool[i+50].intv.first) , _MM_HINT_T0);
                        my_prefetch((const char*)(qbwt.lcpi + tree_pool[i+50].intv.second) , _MM_HINT_T0);
			my_prefetch((const char*)(tree_pool[i + 50].p + tree_pool[i + 50].l - 1) , _MM_HINT_T0);
		    }

		    
		    tree_shrink_batched(qbwt, cnt, td);
		}
	}


}
