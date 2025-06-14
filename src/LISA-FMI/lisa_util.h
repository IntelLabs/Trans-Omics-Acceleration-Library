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
#ifndef LISA_UTIL_H
#define LISA_UTIL_H

#include <vector>
#include <cstring>
#include <string>
#include <tuple>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <utility>
#include <cstdio>
#include <type_traits>
#include <fstream>
#include <array>
#include <initializer_list>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <climits>
#include <map>
#include <future>
#include <omp.h>
#include <array>
#include <immintrin.h>
#include <iostream>
#include <unistd.h>
#include "FMI_search.h"
using namespace std;

#ifndef __rdtsc 
#ifdef _rdtsc
#define __rdtsc _rdtsc
#else
#define __rdtsc __builtin_ia32_rdtsc 
#endif
#endif

#ifdef __lg
#undef __lg
#endif

#ifndef _MM_HINT_NT
#define _MM_HINT_NT _MM_HINT_NTA
#endif 

#ifdef ENABLE_PREFETCH
#define my_prefetch(a, b) _mm_prefetch(a, b)
#else
#define my_prefetch(a, b)
#endif 

template<typename T>
inline constexpr unsigned long __lg(T n) {
    return sizeof(uint64_t) * __CHAR_BIT__  - 1 - __builtin_clzll(n);
}

#define eprintln(...) do{\
    fprintf(stderr,__VA_ARGS__);\
    fprintf(stderr,"\n");\
}while(0)

#define error_quit(...) do{\
    eprintln(__VA_ARGS__);\
    exit(1);\
}while(0)

const string dna = "ACGT";
constexpr int dna_ord(const char &a) {
#ifdef NO_DNA_ORD 
    __builtin_unreachable();
    // assert(0 && "dna_ord is not supported");
#else
    return __lg(a-'A'+2)-1; // "ACGT" -> 0123
#endif 
}

typedef int64_t index_t;

class SMEM_out {
    public:
	    int id, q_l, q_r; 
	    index_t ref_l, ref_r;
	    SMEM_out(int _id, int _q_l, int _q_r, index_t _ref_l, index_t _ref_r);

};

class Output {
	public:
		int id;
		SMEM_out* smem;
		SMEM* tal_smem;
		Output(int a);
};

// Meta-information for a read 
struct Info {
	const char* p; // pointer to read seq
	int l, r; // Tracks left and right position which performing a search
	int len; // length of read sequence
	uint64_t id; // Read ID
	pair<index_t, index_t> intv; //mem2: <l, l+s>
	int min_intv; // Threshold on the SA size, also reused as max_intv
	int mid; // a pivot point for SMEM search

	int prev_l; // Tracks the left position of previously computed SMEM
	void set(int a, int b, index_t c, index_t d);

	uint64_t get_enc_str();
};

//Used for interval tree structure
struct LcpInfo {
	uint16_t s_width: 12;
	uint8_t bw_ext_msk: 4;
};

//ipbwt entry
typedef pair<uint64_t, uint64_t> ipbwt_t;

// SA interval type        
struct Interval{ index_t low, high; }; // left-inclusive

void load(string filename, vector<char*> ptrs, vector<size_t> sizes);

void save(string filename, vector<char*> ptrs, vector<size_t> sizes);
int64_t FCLAMP(double inp, double bound);

string get_abs_path(string path);
string get_abs_location(string path);

#endif
