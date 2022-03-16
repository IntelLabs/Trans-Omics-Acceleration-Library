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
#ifndef QBWT_IPBWT_RMI_H
#define QBWT_IPBWT_RMI_H
#include "lisa_util.h"
#include "thread_data.h"
//#include "chunkEncode.h"
#include "ipbwt_rmi.h"
//#ifdef lisa_fmi
#include "fmi.h"
//#endif


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
	if(q.r > q.mid)\
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
            /*LISA_search<index_t>::LcpInfo* info = s_info[i];*/ \
            LcpInfo* info = s_info[i]; \
            uint8_t &msk = s_msk[i] 


template<typename index_t>
class LISA_search : public FMI_search {
    public:
	LISA_search(){};
        LISA_search(string t, index_t t_size, string ref_seq_filename, int K, int64_t num_rmi_leaf_nodes);
        ~LISA_search();
        pair<int,int>* all_SMEMs(const char* p, const int p_len, pair<int,int>* ans_ptr, const int min_seed_length) const;
        pair<int,int>* print_all_SMEMs(const char* p, const int p_len, pair<int,int>* ans_ptr, const int min_seed_length, const int &shift) const;
        index_t n;
        typedef typename FMI<index_t>::Interval Interval;
        Interval init_intv;
        void forward_step(const char* p, Interval &intv, int &l, int &r) const;
        void backward_step(const char* p, Interval &intv, int &l, int &r) const;

    // private:
#ifdef lisa_fmi
        FMI<index_t> *fmi;
#else    
	FMI_search *fmiSearch;
#endif
        typedef float linreg_t;
        typedef uint64_t kenc_t; 
        IPBWT_RMI<index_t, kenc_t> *rmi;

        static constexpr uint16_t WID_MAX = (1<<12)-1;
        LcpInfo *lcpi;
        vector<pair<index_t, index_t>> b_width;
        uint8_t *lcpp1;
        static constexpr uint8_t LCPP1_MAX = (1<<8)-1;
        vector<pair<index_t, index_t>> large_lcpp1;
        int64_t get_lcp(index_t i) const;
        pair<Interval, index_t> forward_shrink_phase(Interval intv, char a) const;
        pair<index_t, index_t> advance_chunk(kenc_t first, pair<index_t, index_t> intv) const;
        void load(string filename);
        void save(string filename) const;

	// smem computation
	void smem_rmi_batched(Info *qs, int64_t qs_size, int64_t batch_size, threadData &td, Output* output, int min_seed_len, bool apply_lisa = true);

	void s_pb(Info &_q, int cnt, threadData &td);

	void fmi_extend_batched(int cnt, Info* q_batch, threadData &td, Output* output, int min_seed_len);

	void tree_shrink_batched(int cnt, threadData &td);

	void fmi_shrink_batched(int cnt, Info* q_batch, threadData &td, Info* output, int min_seed_len);


	void exact_search_rmi_batched_k3(Info *qs, int64_t qs_size, int64_t batch_size, threadData &td, Output* output, int min_seed_len, int tid = 0);

	int64_t bwtSeedStrategyAllPosOneThread_with_info_prefetch(
                                                   int32_t numReads,
                                                   int32_t minSeedLen,
                                                   SMEM *matchArray,
						   Info* qs, 
						   threadData & td, 
						   int tid = 0);
	SMEM get_info_to_smem(Info q, int64_t rmi_k, int K);
	private:

	void prepareChunkBatch(Info* qPool, int qPoolSize, uint64_t* str_enc, int64_t* intv_all, int K); 
	void prepareChunkBatchForward(Info* qPool, int qPoolSize, uint64_t* str_enc, int64_t* intv_all, int K);
	void prepareChunkBatchForwardComp(Info* qPool, int qPoolSize, uint64_t* str_enc, int64_t* intv_all, int K, int64_t qbwt_n); 

};
template<typename index_t>
int64_t LISA_search<index_t>::get_lcp(index_t i) const {
    return (lcpp1[i] == LCPP1_MAX ?
	    (int64_t)lower_bound(large_lcpp1.begin(), large_lcpp1.end(), i,
		[&](pair<index_t, index_t> p, index_t q){return p.first < q;})->second:
	    (int64_t)lcpp1[i]) - 1;
}

template<typename index_t>
pair<index_t, index_t> LISA_search<index_t>::advance_chunk(kenc_t first, pair<index_t, index_t> intv) const {
    return rmi->backward_extend_chunk(first, {intv.first, intv.second});
}
template<typename index_t>
void LISA_search<index_t>::load(string filename) {
    ifstream instream(filename.c_str(), ifstream::binary);
    instream.seekg(0);

//#ifndef HUGE_PAGE  
    lcpp1 = new uint8_t[n+1]();
//#else
//    lcpp1 = (uint8_t*) mmap(NULL, sizeof(lcpp1[0]) * (n+1), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
//#endif


    eprintln("MEM-SIZE: lcp1 %lld", (long long)sizeof(lcpp1[0])*(n+1));
    instream.read((char*)lcpp1, (n+1)*sizeof(lcpp1[0]));
  
#ifndef HUGE_PAGE  
    lcpi = new LcpInfo[n+1]();
#else
    lcpi = (LcpInfo*) mmap(NULL, sizeof(lcpi[0])*(n+1), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
#endif
    eprintln("MEM-SIZE: lcpi %lld", (long long)sizeof(lcpi[0])*(n+1));
    instream.read((char*)lcpi, (n+1)*sizeof(lcpi[0]));

    int64_t siz;
    instream.read((char*)&siz, sizeof(siz));
    b_width.resize(siz);
    eprintln("MEM-SIZE: b_width %lld", (long long)sizeof(b_width[0])*siz);

#define LOADV(v) do{\
    for(auto itl=v.begin(), itr=itl; itl<v.end(); itl=itr) {\
        static constexpr int Z = 1<<16;\
        itr = min(itl+Z, v.end());\
        static typename std::remove_const<typename std::remove_reference<decltype(*itl)>::type>::type tmp[Z];\
        int len = itr-itl;\
        instream.read((char*)tmp, len*sizeof(tmp[0]));\
        for(int i=0; i<len; i++) itl[i] = tmp[i];\
    }\
}while(0)

    LOADV(b_width);

    instream.read((char*)&siz, sizeof(siz));
//    large_lcpp1.resize(siz);
//    LOADV(large_lcpp1);

    instream.close();
#undef LOADV
}

template<typename index_t>
void LISA_search<index_t>::save(string filename) const {
    ofstream outstream(filename.c_str(), ofstream::binary);
    outstream.seekp(0);

    outstream.write((char*)lcpp1, (n+1)*sizeof(lcpp1[0]));
    
    outstream.write((char*)lcpi, (n+1)*sizeof(lcpi[0]));

    int64_t siz = (int64_t)b_width.size();
    outstream.write((char*)&siz, sizeof(siz));
#define SAVEV(v) do{\
    for(auto itl=v.begin(), itr=itl; itl<v.end(); itl=itr) {\
        static constexpr int Z = 1<<16;\
        itr = min(itl+Z, v.end());\
        static typename std::remove_const<typename std::remove_reference<decltype(*itl)>::type>::type tmp[Z];\
        int len = itr-itl;\
        for(int i=0; i<len; i++) tmp[i] = itl[i];\
        outstream.write((char*)tmp, len*sizeof(tmp[0]));\
    }\
}while(0)

    SAVEV(b_width);

    siz = (int64_t)large_lcpp1.size();
    outstream.write((char*)&siz, sizeof(siz));
    SAVEV(large_lcpp1);

    outstream.close();
#undef SAVEV
}

template<typename index_t>
pair<typename FMI<index_t>::Interval, index_t> LISA_search<index_t>::forward_shrink_phase(Interval intv, char a) const {

    index_t e[2] = {intv.low, intv.high};
    LcpInfo info[2] = {lcpi[e[0]], lcpi[e[1]]};
#define GET_WIDTH_ONE(i) (info[i].s_width == WID_MAX ? \
        (index_t)lower_bound(b_width.begin(), b_width.end(), e[i],  \
            [&](pair<index_t, index_t> p, index_t q){return p.first < q;})->second: \
        (index_t)info[i].s_width)
    index_t siz[2] = {GET_WIDTH_ONE(0), GET_WIDTH_ONE(1)};

#ifndef NO_DNA_ORD 
    const uint8_t msk = 1<<dna_ord(a);
#else 
    const uint8_t msk = 1<<(a);
#endif 
#define CAN_BW_EXTEND(i) (info[i].bw_ext_msk & msk)
    
    bool small;
#define big (!small)
#define FORWARD_SHRINK do{\
        e[small] = (e[small] > e[big] ? e[big] + siz[small] : e[big] - siz[small]);\
        info[small] = lcpi[e[small]];\
        siz[small] = GET_WIDTH_ONE(small);\
    }while(0)

    for(small = siz[1] < siz[0]; !CAN_BW_EXTEND(small); small = siz[1] < siz[0]) {
        FORWARD_SHRINK;
    }

    if(e[small] > e[big]) {
        return {{e[big], e[big] + siz[small]}, get_lcp(e[small])};
    } else {
        return {{e[big]-siz[small], e[big]}, get_lcp(e[small])};
    }
#undef FORWARD_SHRINK
#undef big
#undef CAN_BW_EXTEND
#undef GET_WIDTH_ONE
}

template<typename index_t>
inline void LISA_search<index_t>::forward_step(const char *p, Interval &intv, int &l, int &r) const
{
    index_t siz;
    tie(intv, siz) = forward_shrink_phase(intv, p[l-1]);
    r = l + (int)siz;
}

template<typename index_t>
inline void LISA_search<index_t>::backward_step(const char *p, Interval &intv, int &l, int &r) const
{


    for(int i=l-1; i>=0; i--) {
#ifdef lisa_fmi
        auto next = fmi->backward_extend(intv, p[i]);
        if(next.low >= next.high) break;
        else l=i, intv=next;
#else
        auto next = backwardExt_light({intv.low, intv.high}, p[i]);
        if(next.first >= next.second) break;
        else l=i, intv={next.first, next.second};
#endif
    }

}

template<typename index_t>
pair<int,int>* LISA_search<index_t>::all_SMEMs(const char* p, const int p_len, pair<int,int>* ans_ptr, const int min_seed_length) const {
    Interval intv = init_intv;
    auto end = ans_ptr;
    int l = p_len, r = p_len; // [l,r)
    backward_step(p, intv, l, r);
    if(r-l>=min_seed_length) {
        *(end++) = make_pair(l, r);
    }

    while(l != 0) {
        forward_step(p, intv, l, r);
        backward_step(p, intv, l, r);
        if(r-l>=min_seed_length) {
            *(end++) = make_pair(l, r);
        }
    }

    reverse(ans_ptr, end);
    return end;
}

template<typename index_t>
pair<int,int>* LISA_search<index_t>::print_all_SMEMs(const char* p, const int p_len, pair<int,int>* ans_ptr, const int min_seed_length, const int &shift) const {
    Interval intv = init_intv;
    vector<Interval> vs;
    auto end = ans_ptr;
    int l = p_len, r = p_len; // [l,r)
    backward_step(p, intv, l, r);
    if(r-l>=min_seed_length) {
        *(end++) = make_pair(l, r);
        vs.push_back(intv);
    }

    while(l != 0) {
        forward_step(p, intv, l, r);
        backward_step(p, intv, l, r);
        if(r-l>=min_seed_length) {
            *(end++) = make_pair(l, r);
            vs.push_back(intv);
        }
    }

    reverse(ans_ptr, end);
    reverse(vs.begin(), vs.end());
    for(int i=0; i<(int)vs.size(); i++) {
        printf("+ %ld %ld %lld %lld\n", 
                (long)ans_ptr[i].first+shift, (long)ans_ptr[i].second+shift,(long long) 1LL*vs[i].low, (long long)1LL*vs[i].high);
    }
    return end;
}


template<typename index_t>
LISA_search<index_t>::~LISA_search(){
	eprintln("qbwt rmi deallocated");
#ifdef lisa_fmi
	delete fmi;
#endif
	delete rmi;

	delete lcpp1;

#ifndef HUGE_PAGE
	delete lcpi;
#else
	munmap(lcpi, sizeof(lcpi[0])*(n+1));
#endif

}


template<typename index_t>
LISA_search<index_t>::LISA_search(string t, index_t t_size, string ref_seq_filename, int K, int64_t num_rmi_leaf_nodes):
#ifdef REV_COMP 
    n(2*(index_t)t_size+1),
#else 
    n(t_size+1),
#endif 
    init_intv({0, n}), FMI_search(ref_seq_filename.c_str()) {

#ifdef REV_COMP
    string bin_filename = ref_seq_filename + ".qbwt4.walg.rev_comp";
#else
    string bin_filename = ref_seq_filename + ".qbwt4.walg";
#endif
    string rmi_filename = bin_filename;
    for(const auto &s:{sizeof(index_t)}) {
        bin_filename += string(".") + to_string(s);
    }
    if(ifstream(bin_filename.c_str()).good()) {
        eprintln("Found existing interval tree file %s!!", (char*)bin_filename.c_str());
        // Load interval tree
	load(bin_filename);
#ifdef lisa_fmi
        fmi = new FMI<index_t>(t.c_str(), n, NULL, "@"+dna, bin_filename);
#else
    	fmiSearch = this;
    	load_index_with_rev_complement();
#endif
        rmi = new IPBWT_RMI<index_t, uint64_t>(t, n, rmi_filename, K, num_rmi_leaf_nodes, NULL);
        eprintln("Load successful.");
        eprintln("large lcp size = %lu", large_lcpp1.size());
        eprintln("large lcp space usage = %.6fN", (double)(large_lcpp1.size()*sizeof(large_lcpp1[0])*1.0/n));
        eprintln("large width size = %lu", b_width.size());
        eprintln("large width space usage = %.6fN", (double)(b_width.size()*sizeof(b_width[0])*1.0/n));
        return;
    } else {
        eprintln("No existing %s. Building...", (char*)bin_filename.c_str());
    }


    assert(t.find('@') == string::npos && t.find('$') == string::npos);
    fprintf(stderr, "ref file name for fmi: %s \n", ref_seq_filename.c_str()); 
   
#ifdef REV_COMP
    // appending reverse complement
    for(int64_t i=(index_t)t.size()-1-(t.back()=='@');i>=0;i--) {
        t.push_back(dna[3-(__lg(t[i]-'A'+2)-1)]);
    }
#endif 
    t.push_back('$');

    fprintf(stderr, "%ld: expected %ld: size of t\n", n, t.size());
    assert(n == (index_t)t.size());


    vector<index_t> sa(n);
    if(numeric_limits<index_t>::max() > n && numeric_limits<index_t>::min() < 0) {
        saisxx(t.c_str(), sa.data(), (index_t)n);
    } else {
       // vector<int64_t> _sa(n);
       // saisxx(t.c_str(), _sa.data(), (int64_t)n);
       // for(index_t i=0;i<n;i++) sa[i]=_sa[i];
        saisxx(t.c_str(), sa.data(), (index_t)n);
    }
    

    fprintf(stderr, "ref file name for fmi: %s \n", ref_seq_filename.c_str()); 
	// build rnk = sa^(-1)
    
    // TODO: remove this memory allocation as it is not required for interval tree building
    //rmi = new IPBWT_RMI<index_t, uint64_t>(t, rmi_filename, K, num_rmi_leaf_nodes, sa.data());

    // build lcp
    {
        lcpp1 = new uint8_t[n+1]();
        large_lcpp1.clear();
        auto write_lcp = [&](index_t i, int64_t result) {
            if(result+1 >= LCPP1_MAX) {
                large_lcpp1.push_back({i, (index_t)(result+1)});
                lcpp1[i] = LCPP1_MAX;
            } else {
                lcpp1[i] = (uint8_t)(result+1);
            }
        };
        {
            vector<index_t> rnk(n);
            for(index_t i=0; i<n; i++) rnk[sa[i]]=i;
            for(int64_t i=0, me=0; i<n; i++) {
                if(rnk[i]==0) me=-1;
                else {
                    me = max(decltype(me)(0),me-1);
                    for(const auto mx = max((index_t)i, sa[rnk[i]-1]);
                            mx+me<n && t[i+me]==t[sa[rnk[i]-1]+me]; me++);
                }
                write_lcp(rnk[i], me);
            }
            write_lcp(n, -1);

	    rnk.clear();
	    rnk.shrink_to_fit();
        }
        sort(large_lcpp1.begin(), large_lcpp1.end());

        eprintln("large lcp size = %lu", large_lcpp1.size());
        eprintln("large lcp space usage = %.6fN", (double)(large_lcpp1.size()*sizeof(large_lcpp1[0])*1.0/n));
    

       sa.clear();
       sa.shrink_to_fit();
    }


    // build tree-structure of uni-lcps
    {
        // build order for traversing
        vector<index_t> dec_lcp_order(n-1); {
            int64_t max_lcp = LCPP1_MAX;
            for(auto p: large_lcpp1) max_lcp = max(max_lcp, (int64_t)p.second);
            vector<index_t> cnt(max_lcp+1, 0);
            for(index_t i=1; i<=n-1; i++) cnt[get_lcp(i)]++;
            for(index_t j=1; j<(index_t)cnt.size(); j++) {
                cnt[j] += cnt[j-1];
            }
            for(int64_t i=n-1; i>=1; i--) {
                dec_lcp_order[--cnt[get_lcp(i)]] = i;
            }
            reverse(dec_lcp_order.begin(), dec_lcp_order.end());
            assert(is_sorted(dec_lcp_order.begin(), dec_lcp_order.end(), [&](index_t i, index_t j){return make_pair(get_lcp(i), i) > make_pair(get_lcp(j),j);}));
            eprintln("dec lcp ordering ok.");
        }

        // il, ir for O(1) interval-merge algorithm
        vector<index_t> il(n);
        iota(il.begin(), il.end(), ((index_t)0));
        vector<index_t> ir=il;

        lcpi = new LcpInfo[n+1]();

        lcpi[0].s_width = lcpi[n].s_width = WID_MAX;
        b_width.push_back({0,n+1});
        b_width.push_back({n,n+1});

#ifdef lisa_fmi
    fmi = new FMI<index_t>(t.c_str(), sa.data(), "@"+dna, bin_filename); // do not directly call load/save!
#else
    //fmiSearch = new FMI_search(ref_seq_filename.c_str());
    //fmiSearch->load_index_with_rev_complement();
    fmiSearch = this;
    load_index_with_rev_complement();
#endif

        for(auto itl=dec_lcp_order.begin(), itr=itl; itl!=dec_lcp_order.end(); itl=itr) {
            for(itr++; itr!=dec_lcp_order.end() &&
                       get_lcp(*itr)==get_lcp(*itl) &&
                       il[(*prev(itr))-1] == *itr; itr++);

            if(itr-itl>1) reverse(itl+1, itr);

            for(auto it=itl;it!=itr;it++) {
                const index_t i = *it;
                const index_t l = il[i-1];
                const index_t r = ir[i];
                il[r]=l;
                ir[l]=r;
                auto w = r-l+1;
                if(w>=WID_MAX) {
                    b_width.push_back({i, w});
                    lcpi[i].s_width = WID_MAX;
                } else {
                    lcpi[i].s_width = (uint16_t)w;
                }
                if(it+1==itr) {
                    uint8_t my_mask = 0;
                    for(int j=0;j<(int)dna.size();j++) {
                        
#ifdef lisa_fmi
			auto intv = fmi->backward_extend({l, r+1}, dna[j]);
			if(intv.low < intv.high) my_mask |= 1<<j;
#else
                        auto intv = backwardExt_light({l, r+1}, j);
                        if(intv.first < intv.second) my_mask |= 1<<j;
#endif                   
                    }
                    lcpi[i].bw_ext_msk = my_mask;
                }
            }
        }
        sort(b_width.begin(), b_width.end());
        eprintln("large width size = %lu", b_width.size());
        eprintln("large width space usage = %.6fN", (double)(b_width.size()*sizeof(b_width[0])*1.0/n));
    }

    eprintln("%s build done.", (char*)bin_filename.c_str());
    save(bin_filename);
    eprintln("save done.");
}

template<typename index_t>
void LISA_search<index_t>::smem_rmi_batched(Info *qs, int64_t qs_size, int64_t batch_size, threadData &td, Output* output, int min_seed_len, bool apply_lisa){
	Info *chunk_pool = td.chunk_pool;
	int &chunk_cnt = td.chunk_cnt;

	uint64_t *str_enc = td.str_enc;
	int64_t *intv_all = td.intv_all;

	Info* fmi_pool = td.fmi_pool;
	int &fmi_cnt = td.fmi_cnt;

	Info* tree_pool = td.tree_pool;
	int &tree_cnt = td.tree_cnt;

	int K = this->rmi->K;		

	LISA_search<index_t> &qbwt = *(this);
	
	int64_t next_q = 0;
	while(next_q < qs_size || (chunk_cnt + fmi_cnt ) > 0){
		while(next_q < qs_size && chunk_cnt < batch_size && fmi_cnt < batch_size){
				if(apply_lisa == true && qs[next_q].r >= K){
					chunk_pool[chunk_cnt++] = qs[next_q++];
				}
				else
					fmi_pool[fmi_cnt++] = qs[next_q++];
		}
		
		// process chunk batch
		if(next_q >= qs_size || !(chunk_cnt < batch_size)){
			prepareChunkBatch(chunk_pool, chunk_cnt, str_enc, intv_all, K);

			this->rmi->backward_extend_chunk_batched(&str_enc[0], chunk_cnt, intv_all);
			auto cnt = chunk_cnt;
			chunk_cnt = 0;
			for(int64_t j = 0; j < cnt; j++) {
				Info &q = chunk_pool[j];
				auto next_intv = make_pair(intv_all[2*j],intv_all[2*j + 1]);

				// next state: chunk batching
				if((next_intv.second - next_intv.first) > q.min_intv){//TODO: min_intv_size
					q.intv = next_intv;
					q.l -= K;

					if(q.l >= K)// heuristics: && !(q.intv.second - q.intv.first > 1 && q.intv.second - q.intv.first < 100))	
						chunk_pool[chunk_cnt++] = q;
					else if(q.l > 0){
						fmi_pool[fmi_cnt++] = q;
					}
					else if(q.r - q.l >= min_seed_len && q.l != q.prev_l){

#ifdef OUTPUT
						
						output->tal_smem[td.numSMEMs].rid = q.id;                                                                                               			
						output->tal_smem[td.numSMEMs].m = q.l;                                                                                               			
						output->tal_smem[td.numSMEMs].n = q.r - 1;                                                                                               			
						output->tal_smem[td.numSMEMs].k = q.intv.first;                                                                                               			
						output->tal_smem[td.numSMEMs].l = 0;                                                                                               			
						output->tal_smem[td.numSMEMs].s = q.intv.second - q.intv.first;
						//output->tal_smem[td.numSMEMs].smem_id = q.smem_id;                                                                                               			
						td.numSMEMs++;                                                                                               			
						q.prev_l = q.l;                                                                        			
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

			this->fmi_extend_batched(cnt, &fmi_pool[0], td, output, min_seed_len);	

			cnt = tree_cnt;
			tree_cnt = 0;
			for(int i = 0; i < cnt; i++) {
				auto &q = tree_pool[i];
				this->s_pb(q, i, td);
				my_prefetch((const char*)(this->lcpi + tree_pool[i+50].intv.first) , _MM_HINT_T0);
				my_prefetch((const char*)(this->lcpi + tree_pool[i+50].intv.second) , _MM_HINT_T0);
				my_prefetch((const char*)(tree_pool[i + 50].p + tree_pool[i + 50].l - 1) , _MM_HINT_T0);
			}
			this->tree_shrink_batched(cnt, td);
		}

	}

}
template<typename index_t>
void LISA_search<index_t>::s_pb(Info &_q, int cnt, threadData &td) {
	

    LISA_search<index_t> &qbwt = *(this);

    Info &q = td.tree_pool[cnt]; 
    index_t* siz = td.s_siz[cnt]; 
    LcpInfo* info = td.s_info[cnt]; 
    uint8_t &msk = td.s_msk[cnt]; 
    q = _q;
    info[0] = qbwt.lcpi[q.intv.first]; info[1] = qbwt.lcpi[q.intv.second];
    siz[0] = GET_WIDTH(info[0].s_width, q.intv.first); siz[1] = GET_WIDTH(info[1].s_width, q.intv.second);
#ifndef NO_DNA_ORD
    msk = 1<<dna_ord(q.p[q.l-1]);
#else 
    msk = 1<<q.p[q.l-1];
#endif 
}



template<typename index_t>
void LISA_search<index_t>::fmi_extend_batched(int cnt, Info* q_batch, threadData &td, Output* output, int min_seed_len) {

	LISA_search<index_t> &qbwt = *(this);
	FMI_search* tal_fmi = this;

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
#ifdef lisa_fmi
		my_prefetch((const char*)(qbwt.fmi->occb + ls), _MM_HINT_T0); 
		my_prefetch((const char*)(qbwt.fmi->occb + hs), _MM_HINT_T0); 
#else
              _mm_prefetch((const char *)(&tal_fmi->cp_occ[(q.intv.first) >> CP_SHIFT]), _MM_HINT_T0);
              _mm_prefetch((const char *)(&tal_fmi->cp_occ[(q.intv.second) >> CP_SHIFT]), _MM_HINT_T0);
#endif
                my_prefetch((const char*)(q.p + q.l -  1) , _MM_HINT_T0); 
	}
	
	while(fmi_batch_size > 0) {

		for(int i = 0; i < fmi_batch_size; i++){
			Info &q = q_batch[i];	
			//process one step
			int it = q.l -1;

			if(it >=0 && (int)q.p[it] < 4){ // considering encoding acgt -> 0123			
#ifdef lisa_fmi
				auto next = qbwt.fmi->backward_extend({q.intv.first, q.intv.second}, q.p[it]);
#else
				std::pair<int64_t, int64_t> next_tal = tal_fmi->backwardExt_light( {q.intv.first, q.intv.second}, q.p[it]);
#endif
				
#ifdef lisa_fmi
				if(!((next.high - next.low) > q.min_intv)) { 
#else
				if(!((next_tal.second - next_tal.first) > q.min_intv)) { 
#endif				

					if(q.r - q.l >= min_seed_len && q.l != q.prev_l){
#ifdef OUTPUT
					
						if(q.mid == 0 || q.mid > 0 && q.l <= q.mid)
						{
	
							output->tal_smem[td.numSMEMs].rid = q.id;                                                                                               			
							output->tal_smem[td.numSMEMs].m = q.l;                                                                                               			
							output->tal_smem[td.numSMEMs].n = q.r - 1;                                                                                               			
							output->tal_smem[td.numSMEMs].k = q.intv.first;                                                                                               			
							output->tal_smem[td.numSMEMs].l = 0;                                                                                               			
							output->tal_smem[td.numSMEMs].s = q.intv.second - q.intv.first;
							td.numSMEMs++;                       
							q.prev_l = q.l;                             

						}                                           			
#endif
					}
					tree_pool[tree_cnt++] = q;  //State change
					if(cnt1< cnt) //More queries to be processed?
						q = q_batch[cnt1++]; //direction +
					else
						q = q_batch[--fmi_batch_size];
          			      	my_prefetch((const char*)(q.p + q.l -  1) , _MM_HINT_T0);

				}
				else {
#ifdef lisa_fmi
					q.l = it, q.intv={next.low, next.high};  //fmi-continue
#else
					q.l = it, q.intv={next_tal.first, next_tal.second};  //fmi-continue
#endif
				}
			}	
			else{
					//query finished
					if(q.r - q.l >= min_seed_len && q.l != q.prev_l){
#ifdef OUTPUT
						
						output->tal_smem[td.numSMEMs].rid = q.id;                                                                                               			
						output->tal_smem[td.numSMEMs].m = q.l;                                                                                               			
						output->tal_smem[td.numSMEMs].n = q.r - 1;                                                                                               			
						output->tal_smem[td.numSMEMs].k = q.intv.first;                                                                                               			
						output->tal_smem[td.numSMEMs].l = 0;                                                                                               			
						output->tal_smem[td.numSMEMs].s = q.intv.second - q.intv.first;
						td.numSMEMs++;                                                                                               			
						q.prev_l = q.l;                                                                        			
#endif
					}
					if(cnt1 < cnt) //More queries to be processed?
						q = q_batch[cnt1++];
					else
						q = q_batch[--fmi_batch_size];
                			my_prefetch((const char*)(q.p + q.l -  1) , _MM_HINT_T0);
			}
			static constexpr int INDEX_T_BITS = sizeof(index_t)*__CHAR_BIT__;
			static constexpr int shift = __lg(INDEX_T_BITS);
			auto ls = ((q.intv.first>>shift)<<3), hs = ((q.intv.second>>shift)<<3); 
#ifdef lisa_fmi
			my_prefetch((const char*)(qbwt.fmi->occb + ls + 4), _MM_HINT_T0); 
			my_prefetch((const char*)(qbwt.fmi->occb + hs + 4), _MM_HINT_T0); 
#else
     	               _mm_prefetch((const char *)(&tal_fmi->cp_occ[(q.intv.first) >> CP_SHIFT]), _MM_HINT_T0);
                       _mm_prefetch((const char *)(&tal_fmi->cp_occ[(q.intv.second) >> CP_SHIFT]), _MM_HINT_T0);
#endif
		}
	}
}


template<typename index_t>
void LISA_search<index_t>::tree_shrink_batched(int cnt, threadData &td){

	LISA_search<index_t> &qbwt = *(this);
	Info* tree_pool = td.tree_pool;
	index_t** s_siz = td.s_siz;
	LcpInfo** s_info = td.s_info;
	uint8_t* s_msk = td.s_msk;

	Info* fmi_pool = td.fmi_pool;
	int &fmi_cnt = td.fmi_cnt;

	int pref_dist = 50;
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

template<typename index_t>
void LISA_search<index_t>::fmi_shrink_batched(int cnt, Info* q_batch, threadData &td, Info* output, int min_seed_len){

	LISA_search<index_t> &qbwt = *(this);
	FMI_search* tal_fmi = this;


	Info* tree_pool = td.tree_pool;
	int &tree_cnt = td.tree_cnt;
	tree_cnt = 0;
	int output_cnt = 0;	

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
#ifdef lisa_fmi
		my_prefetch((const char*)(qbwt.fmi->occb + ls), _MM_HINT_T0); 
		my_prefetch((const char*)(qbwt.fmi->occb + hs), _MM_HINT_T0); 
#else
	      _mm_prefetch((const char *)(&tal_fmi->cp_occ[(q.intv.first) >> CP_SHIFT]), _MM_HINT_T0);
              _mm_prefetch((const char *)(&tal_fmi->cp_occ[(q.intv.second) >> CP_SHIFT]), _MM_HINT_T0);
#endif
              my_prefetch((const char*)(q.p + q.l) , _MM_HINT_T0); 
	}
	
	while(fmi_batch_size > 0) {

		for(int i = 0; i < fmi_batch_size; i++){
			Info &q = q_batch[i];	
			//process one step
			int it = q.l;

			if(it < q.r){			
#ifdef lisa_fmi
				auto next = qbwt.fmi->backward_extend({q.intv.first, q.intv.second}, 3 - q.p[it]);
				
#else
				std::pair<int64_t, int64_t> next_tal = tal_fmi->backwardExt_light( {q.intv.first, q.intv.second}, 3 - q.p[it]);
				
#endif
#ifdef lisa_fmi
				if((next.high - next.low) < q.min_intv) { 
#else
				if((next_tal.second - next_tal.first) < q.min_intv) { 
#endif					
					q.r = q.l;	
					output[output_cnt++] = q;  //State change

					if(cnt1< cnt) //More queries to be processed?
						q = q_batch[cnt1++]; //direction +
					else
						q = q_batch[--fmi_batch_size];
          			      	my_prefetch((const char*)(q.p + q.l) , _MM_HINT_T0);

				}
				else {
#ifdef lisa_fmi
					q.l = it + 1, q.intv={next.low, next.high};  //fmi-continue
#else
					q.l = it + 1, q.intv={next_tal.first, next_tal.second};  //fmi-continue
#endif
				}
			}	
			else{
					output[output_cnt++] = q;  //State change
					//query finished
					if(cnt1 < cnt) //More queries to be processed?
						q = q_batch[cnt1++];
					else
						q = q_batch[--fmi_batch_size];
                			my_prefetch((const char*)(q.p + q.l) , _MM_HINT_T0);
			}
			static constexpr int INDEX_T_BITS = sizeof(index_t)*__CHAR_BIT__;
			static constexpr int shift = __lg(INDEX_T_BITS);
			auto ls = ((q.intv.first>>shift)<<3), hs = ((q.intv.second>>shift)<<3); 
#ifdef lisa_fmi
			my_prefetch((const char*)(qbwt.fmi->occb + ls + 4), _MM_HINT_T0); 
			my_prefetch((const char*)(qbwt.fmi->occb + hs + 4), _MM_HINT_T0); 
#else
        	       _mm_prefetch((const char *)(&tal_fmi->cp_occ[(q.intv.first) >> CP_SHIFT]), _MM_HINT_T0);
              	       _mm_prefetch((const char *)(&tal_fmi->cp_occ[(q.intv.second) >> CP_SHIFT]), _MM_HINT_T0);
#endif		
		}
	}
}

template<typename index_t>
void LISA_search<index_t>::exact_search_rmi_batched_k3(Info *qs, int64_t qs_size, int64_t batch_size, threadData &td, Output* output, int min_seed_len, int tid){
	
    uint64_t tim;// = __rdtsc();

	LISA_search<index_t> &qbwt = *(this);
	FMI_search* tal_fm = this;

	Info *chunk_pool = td.chunk_pool;
	int &chunk_cnt = td.chunk_cnt;

	uint64_t *str_enc = td.str_enc;
	int64_t *intv_all = td.intv_all;

	Info* fmi_pool = td.fmi_pool;
	int &fmi_cnt = td.fmi_cnt;


	int K = qbwt.rmi->K;		
	
	
	int64_t next_q = 0;
    	//tim = __rdtsc();
	while(next_q < qs_size || chunk_cnt > 0){
		
		while(next_q < qs_size && chunk_cnt < batch_size){
				//fprintf(stderr,"Here %ld %ld %ld %ld\n", next_q , qs_size , chunk_cnt , batch_size);	
				if(qs[next_q].l <= qs[next_q].len - K){
					chunk_pool[chunk_cnt++] = qs[next_q];
				}
				next_q++;
		}
	
		// process chunk batch
		if(next_q >= qs_size || !(chunk_cnt < batch_size)){
			
			prepareChunkBatchForward(chunk_pool, chunk_cnt, str_enc, intv_all, K);

			qbwt.rmi->backward_extend_chunk_batched(&str_enc[0], chunk_cnt, intv_all);
	

			auto cnt = chunk_cnt;
			chunk_cnt = 0;
			for(int64_t j = 0; j < cnt; j++) {
				Info &q = chunk_pool[j];
				auto next_intv = make_pair(intv_all[2*j],intv_all[2*j + 1]);
				int max_intv = q.min_intv;//TODO: used as max_intv_size here
					

				q.intv = next_intv;

				if((next_intv.second - next_intv.first) < max_intv) { 
					/*&& q.r - q.l >= min_seed_len -- this is always true for K==seed_len*/
					
					q.r = q.l + K - 1;
					if(next_intv.second - next_intv.first > 0){
						
						SMEM s_out;
						s_out.rid = q.id;
						s_out.m = q.l;
						s_out.n = q.r;
						s_out.k = q.intv.first;
						s_out.l = 0;        
						s_out.s = q.intv.second - q.intv.first;                                                                                       			
						output->tal_smem[td.numSMEMs++] = s_out; 

					}
					q.l += K;
					q.intv = {0, qbwt.n};
					if(q.l <= q.len - K)
						chunk_pool[chunk_cnt++] = q;
				}
				else {
					//Next state: fmi procssing
					
					fmi_pool[fmi_cnt++] = q;
				}

			}
		}
		if((next_q >= qs_size && fmi_cnt > 0) || !(fmi_cnt < batch_size)){

		// RMI rev-complemented call to obtain "smem.l"
		prepareChunkBatchForwardComp(fmi_pool, fmi_cnt, str_enc, intv_all, K, qbwt.n);//hardcode
    		qbwt.rmi->backward_extend_chunk_batched(&str_enc[0], fmi_cnt, intv_all);

		td.numSMEMs += bwtSeedStrategyAllPosOneThread_with_info_prefetch(
                                                        fmi_cnt,  
                                                        min_seed_len ,
                                                        &output->tal_smem[td.numSMEMs],
							fmi_pool, td, tid);      
		fmi_cnt = 0; 
		
		}
	
	
	}
    	//	tprof[K3_TIMER][tid] += __rdtsc() - tim; 


}

template<typename index_t>
int64_t LISA_search<index_t>::bwtSeedStrategyAllPosOneThread_with_info_prefetch( int32_t numReads,
                                                  int32_t minSeedLen,
                                                  SMEM *matchArray,
						  Info* qs, threadData &td, int tid)
{
	FMI_search* tal_fmi = this;
	LISA_search<index_t> &qbwt = *(this);

	uint64_t tim;
	int64_t *intv_all = td.intv_all;

	int64_t numTotalSeed = 0;
	int K = qbwt.rmi->K;

	int pref_dist = 30;
	int fmi_batch_size = pref_dist = min(pref_dist, numReads);
	pref_dist = fmi_batch_size;

	SMEM smem_batch[fmi_batch_size];
	Info pf_batch[fmi_batch_size];
	//const char* p_batch[fmi_batch_size];
	

	int next_read_idx = 0;
	int max_intv = qs[0].min_intv;	

	tim = __rdtsc();
	// prepare first batch
	for(int i = 0; i < fmi_batch_size; i++){

		// Forward search
		SMEM smem = get_info_to_smem(qs[i], intv_all[2*i], K);
		smem_batch[next_read_idx] = smem;	
		pf_batch[next_read_idx++] = qs[i];
#ifdef ENABLE_PREFETCH
                        _mm_prefetch((const char *)(&tal_fmi->cp_occ[(smem.k + 4) >> CP_SHIFT]), _MM_HINT_T0);
                        _mm_prefetch((const char *)(&tal_fmi->cp_occ[(smem.k + smem.s + 4) >> CP_SHIFT]), _MM_HINT_T0);
                	_mm_prefetch((const char*)(qs[i].p + qs[i].l + K) , _MM_HINT_T0); 
#endif
	}

	while(fmi_batch_size > 0) {

		for(int i = 0; i < fmi_batch_size; i++)
		{
			// Forward search
			Info &q  = qs[i];
			SMEM &smem = smem_batch[i];
			const char* p = q.p;
			int j = smem.n;
			int readlength = q.len;
			
			if(j < readlength && p[j] < 4)
			{
				smem = tal_fmi->backwardExt(smem, 3 - p[j]);
				if((smem.s < max_intv) /*&& ((smem.n - smem.m + 1) >= minSeedLen)*/) 
				{
					if(smem.s > 0)
					{
						swap(smem.l, smem.k);
						matchArray[numTotalSeed++] = smem;
				    	//	print_smem_lisa(smem);
					}
					// query finished - replace with new one.
					j = smem.n + 1;
					if(j < readlength && readlength - j >= minSeedLen){
						q.l = q.r = j;
						q.intv = {0, qbwt.n};
						td.chunk_pool[td.chunk_cnt++] = q;
					}
					if (next_read_idx < numReads) { 
						q = qs[next_read_idx];
						smem = get_info_to_smem(q, intv_all[2*(next_read_idx)], K);
						next_read_idx++;
					}
					else { 
						q = qs[fmi_batch_size - 1]; 
						smem = smem_batch[fmi_batch_size - 1];
						fmi_batch_size--;
					}
				}
				else smem.n++;
			}
			else {
				j = smem.n + 1;
				if(j < readlength && readlength - j >= minSeedLen){
					q.l = q.r = j;
					q.intv = {0, qbwt.n};
					td.chunk_pool[td.chunk_cnt++] = q;
				}
				if (next_read_idx < numReads) { 
					q = qs[next_read_idx];
					smem = get_info_to_smem(q, intv_all[2*(next_read_idx)], K);
					next_read_idx++;
				}
				else { 
					q = qs[fmi_batch_size - 1]; 
					smem = smem_batch[fmi_batch_size - 1];
					fmi_batch_size--;
				}
			}
#ifdef ENABLE_PREFETCH
                        _mm_prefetch((const char *)(&tal_fmi->cp_occ[(smem.k + 4) >> CP_SHIFT]), _MM_HINT_T0);
                        _mm_prefetch((const char *)(&tal_fmi->cp_occ[(smem.k + smem.s + 4) >> CP_SHIFT]), _MM_HINT_T0);
                	_mm_prefetch((const char*)(q.p + q.l + K) , _MM_HINT_T0); 
#endif
		}

	}
	return numTotalSeed;
}

template<typename index_t>
SMEM LISA_search<index_t>::get_info_to_smem(Info q, int64_t rmi_k, int K){
	SMEM smem;
	smem.rid = q.id;
	smem.m = q.l;
	smem.n = q.l + K;
	smem.l = q.intv.first;
	smem.k = rmi_k;
	smem.s = q.intv.second -  q.intv.first;

	return smem;
}

template<typename index_t>
void LISA_search<index_t>::prepareChunkBatch(Info* qPool, int qPoolSize, uint64_t* str_enc, int64_t* intv_all, int K){

		    for(int64_t j = 0; j < qPoolSize; j++)
		    {
			Info &q = qPool[j];
			uint64_t nxt_ext = 0;
#ifndef NO_DNA_ORD
			for(int i = q.l-K; i != q.l; i++) {
			    nxt_ext = (nxt_ext<<2) | dna_ord(q.p[i]); 
			}
#else			
			for(int i = q.l-K; i != q.l; i++) {
			    nxt_ext = (nxt_ext<<2) | (q.p[i]); 
			}
#endif
			str_enc[j] = nxt_ext;

			intv_all[2 * j] = q.intv.first;
			intv_all[2 * j + 1] = q.intv.second;
			const char *p = qPool[j + 40].p; int offset = qPool[j + 40].l -  K;
                        my_prefetch((const char*)(p + offset) , _MM_HINT_T0);
		    }
}


template<typename index_t>
void LISA_search<index_t>::prepareChunkBatchForwardComp(Info* qPool, int qPoolSize, uint64_t* str_enc, int64_t* intv_all, int K, int64_t qbwt_n){

	for(int64_t j = 0; j < qPoolSize; j++)
	{
		Info &q = qPool[j];
		uint64_t nxt_ext = 0;
#ifndef NO_DNA_ORD
		for(int i = q.l + K - 1; i >= q.l; i--) {
			int base = 3 - dna_ord(q.p[i]);
			nxt_ext = (nxt_ext<<2) | base; 
		}
#else			
		for(int i = q.l + K - 1; i >= q.l; i--) {
			int base = 3 - q.p[i];
			nxt_ext = (nxt_ext<<2) | base; 
		}
#endif
		str_enc[j] = nxt_ext;

		intv_all[2 * j] = 0;
		intv_all[2 * j + 1] = qbwt_n;
		const char *p = qPool[j + 40].p; int offset = qPool[j + 40].l;
		my_prefetch((const char*)(p + offset) , _MM_HINT_T0);
	}
}

template<typename index_t>
void LISA_search<index_t>::prepareChunkBatchForward(Info* qPool, int qPoolSize, uint64_t* str_enc, int64_t* intv_all, int K){

	for(int64_t j = 0; j < qPoolSize; j++)
	{
		Info &q = qPool[j];
		uint64_t nxt_ext = 0;
#ifndef NO_DNA_ORD
		for(int i = q.l; i < q.l + K; i++) {
			int base = dna_ord(q.p[i]);
			nxt_ext = (nxt_ext<<2) | base; 
		}
#else			
		for(int i = q.l; i < q.l + K; i++) {
			int base = q.p[i];
			nxt_ext = (nxt_ext<<2) | base; 
		}
#endif
		str_enc[j] = nxt_ext;

		intv_all[2 * j] = q.intv.first;
		intv_all[2 * j + 1] = q.intv.second;
		const char *p = qPool[j + 40].p; int offset = qPool[j + 40].l;
		my_prefetch((const char*)(p + offset) , _MM_HINT_T0);
	}
}

#endif
