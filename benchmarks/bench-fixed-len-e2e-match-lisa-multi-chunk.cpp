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

#include<fstream>
#include "lisa_util.h"
#include "read.h"
#include <immintrin.h>
#include "sais.h"
#include "ipbwt_rmi.h"
#include <omp.h>

#define QUERY_DB_SIZE 12500000000L
unsigned char nst_nt4_table_1[256] = {
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 5 /*'-'*/, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  3, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  3, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4
};

void encode_read(uint8_t *enc_qdb,
				 bseq1_t *seqs,
				 int64_t numQueries, uint32_t query_length,
				 uint64_t start, uint64_t &last, uint64_t max_num_queries, vector <uint64_t*> &chunk_list, int K){

		uint64_t orig_query_count = start + (uint64_t) numQueries;

		if(orig_query_count > max_num_queries)
			orig_query_count = max_num_queries;
	numQueries = 0;
	for (int st=start; st < orig_query_count; st++) {
        int cind = numQueries * query_length;
        int nflag = 0;
        for(int r_chunk = 0; (r_chunk + K - 1) < query_length; r_chunk = r_chunk + K) {
			uint64_t *lisa_enc = chunk_list[r_chunk/K];
			lisa_enc[st] = 0;
			uint64_t enc_val = 0;
			for ( int r = r_chunk; r < r_chunk + K; ++r) {
				uint8_t base = nst_nt4_table_1[seqs[st].seq[r]];
				enc_val = (enc_val<<2) | (uint64_t) base; //enc_qdb[r+cind];
			
        		if (base > 3) nflag = 1;
			}
			lisa_enc[st] = enc_val;
        }
		if(nflag == 0)
            numQueries++;
    }
	
	last = start + numQueries;
}


// Globals for collecting profiling data
int64_t one_calls = 0;
int64_t bin_search_walk = 0;

int main(int argc, char** argv) {
#ifdef VTUNE_ANALYSIS
    __itt_pause();
#endif
	uint64_t tim = __rdtsc();
    sleep(1);
    uint64_t proc_freq = __rdtsc() - tim;
	eprintln("proc freq = %lu", proc_freq);

    if(!(argc == 5 || argc == 6)) {
        error_quit("Need 5 args: ref_file query_set K num_rmi_leaf_nodes num_threads");
    }

    int K = atoi(argv[3]);
    eprintln("using K = %d", K);
    if(K < 0 || K > 30) return 0;

    int64_t num_rmi_leaf_nodes = atol(argv[4]);
    eprintln("using num_rmi_leaf_nodes = %ld", num_rmi_leaf_nodes);
    int numThreads = atoi(argv[5]);



// ----------------------- Step 1: Reference sequence : file reading --------------------------

    string seq;// = read_seq(argv[1]);
    read_seq_lisa(argv[1], seq);
    eprintln("Read ref file done.");

// ----------------------- Step 1 end -----------------------------------------------------


// ----------------------- Step: Input Reads: file reading ----------------------------------
    eprintln("seq.size() = %lu", seq.size());
    string queries; int max_query_len = 0;
    // TODO: Remove this and replace by TAL's file reading style
	tie(queries, max_query_len) = read_query_separated_with_dot(argv[2]);
    eprintln("Read query file done.");

	// TAL style fastq reading
    int32_t numQueries = 0;
    int64_t total_size = 0;
    gzFile fp = gzopen(argv[2], "r");
    bseq1_t *seqs = bseq_read_one_fasta_file(QUERY_DB_SIZE, &numQueries, fp, &total_size);
    
	if(seqs == NULL)
    {
        fprintf(stderr,"ERROR! seqs = NULL\n");
        exit(EXIT_FAILURE);
    }
    int max_query_length = seqs[0].l_seq;
    int min_query_length = seqs[0].l_seq;
    int max_len_id = 0;
    int min_len_id = 0;
    for(int i = 1; i < numQueries; i++)
    {
        if(max_query_length < seqs[i].l_seq)
        {
            max_query_length = seqs[i].l_seq;
            max_len_id = i;
        }
        if(min_query_length > seqs[i].l_seq)
        {
            min_query_length = seqs[i].l_seq;
            min_len_id = i;
        }
    }
    fprintf(stderr,"max_query_length = %d, min_query_length = %d\n", max_query_length, min_query_length);
    fprintf(stderr,"max_len_id = %d, min_len_id = %d\n", max_len_id, min_len_id);
    assert(max_query_length > 0);
    assert(max_query_length < 10000);
    assert(max_query_length == min_query_length);
    assert(numQueries > 0);
    assert(numQueries * (int64_t)max_query_length < QUERY_DB_SIZE);
    int32_t query_length = max_query_length;
    fprintf(stderr,"numQueries = %d, query_length = %d\n", numQueries, query_length);

	int num_chunks = query_length / K;

	assert(query_length % K == 0);
	vector<uint64_t*> chunk_list;
	for(int i = 0; i < num_chunks; i++) {	
    	uint64_t *str_enc = (uint64_t *)malloc(numQueries * sizeof(uint64_t));
		chunk_list.push_back(str_enc);
	}

#ifndef PAR_ENC
	uint64_t filtered_num_queries = 0;
//	encode_read(NULL, seqs, numQueries, query_length, 0 , filtered_num_queries, numQueries, chunk_list[0], K);
	encode_read(NULL, seqs, numQueries, query_length, 0 , filtered_num_queries, numQueries, chunk_list, K);
#endif
	//for(int j = 0; j < numQueries; j++){
	//	for(int i = 0; i < chunk_list.size(); i++){
	//		fprintf(stderr," %d %d %lu\n", i, j, chunk_list[i][j]);
	//	}
	//}

// ------------------------Step 2 end ------------------------------------------------------------


// ------------------------- Index loading -------------------------------------
#ifdef REV_COMP
    eprintln("No char placed between ref seq and reverse complement, to replicate BWA-MEM bug.");
    // appending reverse complement
    for(int64_t i=(index_t)seq.size()-1-(seq.back()=='@');i>=0;i--) {
#ifndef NO_DNA_ORD 
        seq.push_back(dna[3-dna_ord(seq[i])]);
#else 
        seq.push_back(dna[3-(__lg(seq[i]-'A'+2)-1)]);
#endif 
    }
#endif 

    seq.push_back('$');


string ref_seq_filename = argv[1];
#ifdef REV_COMP
    string rmi_filename = ref_seq_filename + ".rev_comp";
#else
    string rmi_filename = ref_seq_filename ;
#endif

    IPBWT_RMI<index_t, uint64_t> rmi(seq, seq.size(), rmi_filename, K, num_rmi_leaf_nodes, NULL);

// ---------------------------Index loading done -----------------------------------------------------

    int64_t numMatches = 0;
    int64_t totalTicks = 0;

    int64_t q_size = numQueries;
    int64_t *intv_all = (int64_t *)malloc(numQueries * 2 * sizeof(int64_t));

    //assert(str_enc != NULL && intv_all != NULL);


  #pragma omp parallel num_threads(numThreads)
  {
	int id = omp_get_thread_num();
	if(id == 0)
		eprintln("Thread created");

  }
  for(int64_t j = 0; j < numQueries; j++)
  {
        intv_all[2 * j] = 0;
        intv_all[2 * j + 1] = rmi.n;
  }
#ifdef VTUNE_ANALYSIS
        __itt_resume();
#endif

//totalTicks -= __rdtsc();
uint64_t matchCount = 0;
int optimal_num_threads = numThreads; //min(34, numThreads);


totalTicks -= __rdtsc(); 
//while(num_iter >= 0)
//{
	//uint64_t *str_enc = chunk_list[num_iter];

	#if ENABLE_PREFETCH
	int64_t parallel_batch_size  = ceil((q_size/numThreads + 1)/80);
	#pragma omp parallel num_threads(numThreads)
	{
		int64_t workTicks = 0;
		int64_t q_processed = 0;	
		#pragma omp for schedule(dynamic, 1) 
		for(int64_t i = 0; i < q_size; i = i + parallel_batch_size){
			//int tid = omp_get_thread_num();			
			int64_t qs_sz = ((i + parallel_batch_size) <= q_size)? parallel_batch_size: q_size - i;

#ifdef PAR_ENC
			uint64_t filtered_num_queries = 0;
			encode_read(NULL, seqs, qs_sz, query_length, i, filtered_num_queries, numQueries, chunk_list, K);
			//rmi.backward_extend_chunk_batched(&str_enc_v2[i], qs_sz, &intv_all[i*2]); 
#endif
			int num_iter = num_chunks - 1;
			while(num_iter >= 0) // run multiple chunks right to left
			{
				uint64_t *str_enc = chunk_list[num_iter];
				rmi.backward_extend_chunk_batched(&str_enc[i], qs_sz, &intv_all[i*2]); 
				num_iter--;
			}
			q_processed += qs_sz;
			//fprintf(stderr, "processed: %ld \n", q_processed);
		}
	}
	#else
    #pragma omp parallel for num_threads(numThreads)
    for(int64_t i = 0; i < q_size; i++)
    {
            auto q = qs[i];
            auto q_intv = rmi.backward_extend_chunk(str_enc[i], {intv_all[2 * i], intv_all[2 * i + 1]});
    
		    intv_all[2 * i] = q_intv.first;
		    intv_all[2*i + 1] = q_intv.second;
    }
#endif
	//num_iter--;
//}

totalTicks += __rdtsc();

#ifdef VTUNE_ANALYSIS
        __itt_pause();
#endif
        for(int64_t i = 0; i < q_size; i++)
        {
            int64_t nm = intv_all[2 * i + 1] - intv_all[2 * i];
            if(nm > 0)
            {
                numMatches += nm;
				matchCount++;
            }
        }

// ----------------- Logging -------------------------- 
    int64_t num_queries = count(queries.begin(), queries.end(), ';');
    assert(num_queries > 0);
    eprintln("Search Done.");
    eprintln("Number of exact matchs = %lld, match count %lld num queries %lld qs_size %lld", (long long)numMatches, (long long)matchCount, (long long)num_queries, (long long)q_size);
    eprintln("total cycle = %lld", (long long)totalTicks);
    eprintln("Ticks per query = %.3f", (double)(totalTicks * 1.0 / num_queries));
    eprintln("%lld: Binary search per query = %.3f", num_rmi_leaf_nodes,(double)(bin_search_walk * 1.0 / num_queries));

#ifdef PRINT_OUTPUT
        for(int64_t i = 0; i < q_size; i++)
        {
            int64_t nm = intv_all[2 * i + 1] - intv_all[2 * i];
            printf("%ld: ",i);
            if(nm > 0)
            {
            	printf("%ld, %ld", intv_all[2 * i], intv_all[2 * i + 1]);
            }
          printf("\n");
        }
#endif
    //free(str_enc); 
	free(intv_all);
    return 0;
}
#undef flip
#undef rev_comp

